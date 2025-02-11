syms x y z

% f1_sym = - exp( - ((x - 1) ^ 2) );
% f1_d1_sym = diff(f1_sym);
% f1_d2_sym = diff(f1_d1_sym)
% f1 = @(x) subs(f1_sym, x); 
% x0 = 0;
% 
% fminsearch(f1, x0)

% f2_sym = (1 - x)^2 + 100 * (y - x^2)^2;
% f2_grad_sym = gradient(f2_sym, [x,y])
% f2_hess_sym = hessian(f2_sym, [x, y])

% f3_sym = [x, y] * [3, 1; 1, 3] * [x; y] + [-1, 1] * [x; y];
% f3_grad_sym = gradient(f3_sym, [x, y])
% f3_hess_sym = hessian(f3_sym, [x, y])


f4_sym = 0.5 * [x y z] * [1 1 0; 1 1 0; 0 0 4] * [x; y; z] - [0 0 1] * [x; y; z];
f4_grad_sym = gradient(f4_sym, [x, y, z]);
f4_hess_sym = hessian(f4_sym, [x, y, z]);
f4 = @(x) 0.5 * [x(1) x(2) x(3)] * [1 1 0; 1 1 0; 0 0 4] * [x(1); x(2); x(3)] - [0 0 1] * [x(1); x(2); x(3)];
x0 = [-10, -10, -10];
fminsearch(f4, x0)

