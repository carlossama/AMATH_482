clear all; close all;
%% Read in the Data

Vid = VideoReader('monte_carlo_low.mp4');

frames = round(Vid.Duration * Vid.FrameRate);

X = zeros(Vid.width * Vid.height, frames);
for i = 1:frames
    X(:,i) = reshape((double(rgb2gray(readFrame(Vid)))), Vid.width * Vid.height, 1);
end
%%
dt = 1/Vid.FrameRate;
t = dt*(1:frames);

%% DMD Calculations

X1 = X(:,1:end-1);
X2 = X(:,2:end);

[U, Sig, V] = svd(X1, 'econ');
S = U'*X2*V*diag(1./diag(Sig));
[eigVec, eigVal] = eig(S);
mu = diag(eigVal);


w = log(mu)/dt;
%%
plot(abs(w));
%%

%Phi = U*eigVec;
clearvars U
Phi = X2 * V / Sig * eigVec;
y_init = Phi \ X1(:,1);

u_modes = zeros(length(y_init),length(t));
for i = 1:length(t)
   u_modes(:,i) = y_init.*exp(w*t(i)); 
end

%%
u_dmd = Phi*u_modes;
background = y_init.'.*Phi*exp(w.*t);
clearvars V  X1 X2
 %% Plot DMD modes
% close all
figure
%background = b.'.*Phi(:,191)*exp(omega.*t);
for i = 1:378
    
    pcolor(flipud(reshape(abs(Xdmd(:,i)),  Vid.height, Vid.width))); colormap gray, shading interp;
    drawnow;    
    
end

%%


 background = zeros(size(Phi));
backgroundIdx = find(abs(w) < 1);
background(:,backgroundIdx) = u_dmd(:,backgroundIdx);
%%

R = background;
R(R>0) = 0;

background = background - R;

Xsparse = X - abs(background);
Xsparse = Xsparse - R;
%%
close all
for i = 1:length(backgroundIdx)
    pcolor(flipud(reshape(abs(Phi(:,backgroundIdx(i))),  Vid.height, Vid.width))); colormap gray, shading interp;
    drawnow;    
end

















[U, S, V] = svd(X1, 'econ');
%%
r = 378;
r = min(r, size(U,2));
U_r = U(:, 1:r); % truncate to rank-r
S_r = S(1:r, 1:r);
V_r = V(:, 1:r);
Atilde = U_r' * X2 * V_r / S_r; % low-rank dynamics
[W_r, D] = eig(Atilde);
%%
Phi = X2 * V_r / S_r * W_r; % DMD modes
lambda = diag(D); % discrete-time eigenvalues
omega = log(lambda)/dt; % continuous-time eigenvalues
%% Compute DMD mode amplitudes b
x1 = X1(:, 1);
b = Phi\x1;
%% DMD reconstruction
mm1 = size(X1, 2); % mm1 = m - 1
time_dynamics = zeros(r, mm1);
t = (0:mm1-1)*dt; % time vector
for iter = 1:mm1
time_dynamics(:,iter)=(b.*exp(omega*t(iter)));
end
Xdmd = Phi * time_dynamics;

%% 
% 
% background = zeros(size(Phi));
% backgroundIdx = find(abs(w) < 1);
% background(:,backgroundIdx) = u_dmd(:,backgroundIdx);
% %%
% 
% R = background;
% R(R>0) = 0;
% 
% background = background - R;
% 
% Xsparse = X - abs(background);
% %Xsparse = Xsparse; % - R;
% %%
% close all
% for i = 1:length(backgroundIdx)
%     pcolor(flipud(reshape(abs(Phi(:,backgroundIdx(i))),  Vid.height, Vid.width))); colormap gray, shading interp;
%     drawnow;    
% end
% %%
% close all
% pcolor(flipud(reshape(abs(background(:,backgroundIdx)),  Vid.height, Vid.width))); colormap gray, shading interp;
%     drawnow;  
