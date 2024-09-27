clear

syms x y z
% f2 = ((1 - x)^2) + (100 * ((y - (x^2))^2));
% dfdx = diff(f, x);
% dfdy = diff(f, y);
% ddfdxx = diff(dfdx, x);
% ddfdxy = diff(dfdx, y);
% ddfdyx = diff(dfdy, x);
% ddfdyy = diff(dfdy, y);
% hessian_f = [ddfdxx, ddfdxy; ddfdyx, ddfdyy];
% xS_yS = [1, 1];
% hessian_f_xS_yS = subs(hessian_f, [x y], xS_yS);
% [eVec, eVal] = eig(hessian_f_xS_yS)

% f3 = 20*x + 2*x^2 + 4*y - 2*y^2;
% df3dx = diff(f3, x)
% df3dy = diff(f3, y)
% h_f3 = hessian(f3);
% xs_ys = [-5, 1];
% [V, D] = eig(subs(h_f3, [x,y], xs_ys))
X2 = [x;y];
% 
% A = [3, 1; 1, 3];
% b = [-1; 1];
% f4 = (X.' * A * X) + (b' * X)
% df4dx = diff(f4, x)
% df4dy = diff(f4, y)
% h_f4 = hessian(f4)
% xs_ys = [0.25, -0.25];
% [V, D] = eig(subs(h_f4, [x,y], xs_ys))

% A = [1, 2; 2, 1];
% b = [1; 10];
% f5 = (X.' * A * X) + (b' * X)
% df5dx = diff(f5, x)
% df5dy = diff(f5, y)
% h_f5 = hessian(f5)
% xs_ys = [0.25, -0.25];
% [V, D] = eig(subs(h_f5, [x,y], xs_ys))


X3 = [x;y;z]
A = [1, 1, 0; 1, 1, 0; 0,0,4];
b = [0;0;2];
f6 = (X3.' * A * X3) + (b' * X3)
df6dx = diff(f6, x)
df6dy = diff(f6, y)
df6dz = diff(f6, z)
h_f6 = hessian(f6)
xS = linsolve(A, b)
[V, D] = eig(subs(h_f6, [x;y;z], xS))