def create_index_to_coord_map():
    """Create a mapping of each index to its x,y coordinate"""
    # Initialize empty dictionary
    index_to_coord = {}
    
    # Create mapping based on the grid
    coord_ranges = {
        (0, 6): [(0, 46), (7247, 7261), (7369, 7451)],
        (-1, 6): [(706, 744), (7276, 7355), (7262, 7275), (7356, 7368)],
        (-2, 6): [(745, 799)],
        (-3, 6): [(888, 929)],
        (-4, 6): [(930, 969)],
        (-5, 6): [(970, 990)],
        
        (0, 7): [(609, 630)],
        (-1, 7): [(631, 652)],
        (-2, 7): [(653, 705)],
        (-3, 7): [(808, 845)],
        (-4, 7): [(848, 887)],
        
        (0, 8): [(463, 508)],
        (-2, 8): [(1167, 1185)],
        (-3, 8): [(1111, 1166)],
        (-4, 8): [(991, 1008)],
        
        (0, 9): [(409, 462)],
        (-2, 9): [(1236, 1268)],
        (-3, 9): [(1269, 1297)],
        (-4, 9): [(1041, 1110)],
        (-5, 9): [(1009, 1040)],
        
        (0, 10): [(1497, 1520)],
        (-2, 10): [(6373, 6438)],
        (-3, 10): [(6355, 6372)],
        (-4, 10): [(1326, 1339), (1438, 1464), (6306, 6354)],
        (-5, 10): [(1340, 1437)],
        
        (0, 11): [(1561, 1605), (6686, 6734)],
        (-1, 11): [(6597, 6615)],
        (-2, 11): [(5806, 5838), (6460, 6478)],
        (-3, 11): [(5839, 5884)],
        (-4, 11): [(6107, 6141)],
        (-5, 11): [(6142, 6178)]
    }
    
    # Fill the dictionary with all index mappings
    for coord, ranges in coord_ranges.items():
        for start, end in ranges:
            for index in range(start, end + 1):
                index_to_coord[index] = coord
    
    return index_to_coord

def get_coordinate(index, index_map):
    """Query an index to get its coordinates"""
    if index in index_map:
        return index_map[index]
    return None

# Example usage
def main():
    # Create the mapping once
    index_map = create_index_to_coord_map()
    
    while True:
        try:
            # Get input from user
            user_input = input("Enter an index (0-7451) or 'q' to quit: ")
            
            # Check for quit command
            if user_input.lower() == 'q':
                break
            
            # Convert input to integer
            index = int(user_input)
            
            # Check if index is in valid range
            if 0 <= index <= 7451:
                coord = get_coordinate(index, index_map)
                if coord:
                    print(f"{index} = ({coord[0]},{coord[1]})")
                else:
                    print(f"No coordinate mapping found for index {index}")
            else:
                print("Index must be between 0 and 7451")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

if __name__ == "__main__":
    main()