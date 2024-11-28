from vis_nav_game import Player, Action
import pygame
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import euclidean_distances
from joblib import load
import re
import os
from maze_solver.maze_solver_da import maze_solver
from matplotlib.pyplot import imsave, imshow, figure, show, waitforbuttonpress
from time import strftime
import pickle


class KeyboardPlayer(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.feature_extractor = None
        self.stored_features = None
        self.image_paths = None
        self.goal = None
        self.goal_loc=None
        self.curr=None
        self.loc_cod=None
        self.matches=[]
        self.maze_img=cv2.imread('maze_solver\\maze.jpg',cv2.IMREAD_GRAYSCALE)
        self.FEATURES_PATH = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\Midterm_project\\feature.joblib\\image_features.joblib"
        self.PATHS_PATH = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\Midterm_project\\feature.joblib\\image_paths.joblib"
        super(KeyboardPlayer, self).__init__()
        try:
            base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
            self.feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
            self.stored_features = load(self.FEATURES_PATH)
            self.image_paths = load(self.PATHS_PATH)
            self.loc_cod =pickle.load(open("location_coord.pkl", "rb"))
            print("Feature matching system initialized successfully")
        except Exception as e:
            print(f"Error initializing feature matching system: {str(e)}")
        

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            # pygame.K_ESCAPE: Action.QUIT
        }
        
        # Load features
        

    def extract_features(self, image_array):
        # Convert to PIL Image and resize
        image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        image = image.resize((224, 224))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        features = self.feature_extractor.predict(image_array, verbose=0)
        return features.flatten().astype(np.float32)

    def find_closest_match(self, target_features):
        distances = euclidean_distances([target_features], self.stored_features)[0]
        closest_match = np.argmin(distances)
        distance=distances.min()
        return closest_match, distance

    def extract_image_number(self, path):
        match = re.search(r'image_(\d+)', path)
        if match:
            return int(match.group(1))
        return None

    def find_matching_image(self,img):
        if img is None:
            print("No current view available")
            return

        try:
            target_features = self.extract_features(img)
            
            closest_index, dist= self.find_closest_match(target_features)
            
            # Get the image number from the matching path
            matching_path = self.image_paths[closest_index]
            image_number = self.extract_image_number(matching_path)
            # print(f"dist {closest_index} image no {image_number}")
            if self.goal is None and image_number is not None and len(self.matches)<=4:
                # self.goal = image_number
                # print(f'Target image id: {self.goal}')
                
                if len(self.matches)==4:
                    self.goal = image_number
                    # print(f'Target image id: {self.goal}')
                    cv2.imshow('Matching Image', cv2.imread(matching_path))
                    cv2.waitKey(1)
                    return
                return matching_path
            
            else:
                if image_number is not None:
                    self.curr=image_number
                    print(f"Current: {image_number} Target: {self.goal}")
                    try:
                        matching_image = cv2.imread(matching_path)
                        if matching_image is not None:
                            color = (0, 0, 0)
                            w_offset = 50
                            h_offset = 50
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            line = cv2.LINE_AA
                            size = 0.75
                            stroke = 2
                            x=(self.fpv-matching_image).mean()
                            if x<60:
                                cv2.putText(matching_image, f'Good Match {x}', (h_offset, w_offset), font, size, color, stroke, line)
                                cv2.imshow('Matching Image', matching_image)
                                cv2.waitKey(1)
                            else:
                                cv2.putText(matching_image, f'BAD Match {x}', (h_offset, w_offset), font, size, color, stroke, line)
                                cv2.imshow('Matching Image', matching_image)
                                cv2.waitKey(1)
                            # Maze code call here just to show current location on map
                            # start_point = (140, 260)
                            # # self.goal_loc = (int(self.loc_cod[self.goal][0]),int(self.loc_cod[self.goal][1]))
                            # maze_solver_obj = maze_solver(src=start_point, dst=self.goal_loc, img=self.maze_img)
                            # p = maze_solver_obj.find_shortest_path()
                            # maze_solver_obj.drawPath(path=p, thickness=2)
                            # cv2.imshow('Maze_Solve',maze_solver_obj.img)
                            # cv2.waitKey(1)
                            # figure(figsize=(7, 7))
                            # imshow(maze_solver_obj.img)
                            # show()
                            # imsave(f'maze_solver\img\maze_solved_{strftime("%Y%m%d_%H%M%S")}.png', maze_solver_obj.img)
                            # cv2.namedWindow('solved maze', cv2.WINDOW_NORMAL) 
                            # cv2.resizeWindow('solved maze', maze_solver_obj.img.shape[0]*2, maze_solver_obj.img.shape[1]*2) 
                            # cv2.imshow(winname='solved maze', mat=maze_solver_obj.img)
                            # waitforbuttonpress(1)

                    except Exception as e:
                        print(f"Could not load matching image: {str(e)}")
                else:
                    print(f"Error: Could not extract image number from path: {matching_path}")
                
        except Exception as e:
            print(f"Error in image matching: {str(e)}")

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                elif event.key == pygame.K_q:  #q for query
                    self.find_matching_image(self.fpv)
                elif event.key==114:
                    self.find_matching_image(self.fpv)
                    maze_solver_obj = maze_solver(src=(int(self.loc_cod[self.curr][0]),int(self.loc_cod[self.curr][1])), dst=self.goal_loc, img=self.maze_img)
                    p = maze_solver_obj.find_shortest_path()
                    maze_solver_obj.drawPath(path=p, thickness=2)
                    cv2.imshow('Maze_Solve',maze_solver_obj.img)
                    cv2.waitKey(1)
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        # match_dist=[]
        if self.goal is None:
            # self.find_matching_image(targets[1])
            for i in range(len(targets)):
                self.find_matching_image(targets[i])
                # self.matches.append()
                # print(self.matches)
                match_im=cv2.imread(self.find_matching_image(targets[i]))
                self.matches.append((match_im-targets[i]).mean())
                # print(self.matches)                    
            self.find_matching_image(targets[self.matches.index(min(self.matches))])
            #Maze Code call Here target id = self.goal start position default
            start_point = (140, 260)
            self.goal_loc = (int(self.loc_cod[self.goal][0]),int(self.loc_cod[self.goal][1])) #(30,24) # TBD Based on indexing code
            # end_point=(30,24)
            print(f"Goal Coordinate : {self.goal_loc}")
            maze_solver_obj = maze_solver(src=start_point, dst=self.goal_loc, img=self.maze_img)
            p = maze_solver_obj.find_shortest_path()
            maze_solver_obj.drawPath(path=p, thickness=2)
            cv2.imshow('Maze_Solve',maze_solver_obj.img)
            cv2.waitKey(1)

        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, f'Front View {str(self.goal)}', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.imwrite('target.jpg', concat_img)
        cv2.waitKey(1)


    def set_target_images(self, images):
        super(KeyboardPlayer, self).set_target_images(images)
        self.show_target_images()

    def pre_exploration(self):
        K = self.get_camera_intrinsic_matrix()
        print(f'K={K}')

    def pre_navigation(self) -> None:
        pass

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            opencv_image = opencv_image[:, :, ::-1] 
            shape = opencv_image.shape[1::-1]
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    import logging
    logging.basicConfig(filename='vis_nav_player.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    import vis_nav_game as vng
    logging.info(f'player.py is using vis_nav_game {vng.core.__version__}')
    vng.play(the_player=KeyboardPlayer())