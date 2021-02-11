clear all; close all; clc
load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; % scale frequencies to the period of the spatial domain (1 = period length...)
ks = fftshift(k);
[X,Y,Z]=meshgrid(x,y,z);
[Kxs,Kys,Kzs]=meshgrid(ks,ks,ks);
[Kx,Ky,Kz]=meshgrid(k,k,k);

Unt_avg = zeros(n,n,n);
for j = 1:49
Un(:,:,:)= reshape(subdata(:,j),n,n,n);
Unt = fftn(Un);
Unt_avg = Unt_avg + Unt;
end
Unt_avgs = abs(fftshift(Unt_avg)) / 49;
M = max(Unt_avgs,[],'all');
[a, b, c] = ind2sub([n,n,n], find(Unt_avgs == M));
M_idx = [a, b, c];

figure(1)
hold on
[r,c,v] = ind2sub(size(Unt_avgs),find(Unt_avgs > 75));
%scatter3(ks(c),ks(r),ks(v),'o','filled')
scatter3(ks(M_idx(2)), ks(M_idx(1)), ks(M_idx(3)),'o','r','filled')
xlabel('x')
ylabel('y')
zlabel('z')

p = patch(isosurface(Kxs,Kys,Kzs,Unt_avgs/M, 0.2));
p.FaceColor = 'cyan';
p.EdgeColor = 'none';
p.FaceAlpha = 0.1;

tau = 1; 
filter = exp(-tau*((Kx - ks(M_idx(2))).^2+(Ky - ks(M_idx(1))).^2+(Kz - ks(M_idx(3))).^2));

p1 = patch(isosurface(Kxs,Kys,Kzs,fftshift(filter), 0.2));
p1.FaceColor = 'green';
p1.EdgeColor = 'none';
p1.FaceAlpha = 0.5;

traj = zeros(49,3);
for j = 1:49
Un(:,:,:)= reshape(subdata(:,j),n,n,n);
Unt =  fftn(Un);
Untf = Unt.*filter;
Unf = ifftn(Untf);

val = max(abs(Unf),[],'all');
[a, b, c] = ind2sub([n,n,n], find(abs(Unf) == val));
traj(j,:) = [x(a), y(b), z(c)];

end
figure(2)
plot3(traj(:,1),traj(:,2),traj(:,3),'o-');
xlabel('x axis')
ylabel('y axis')
zlabel('z axis')
xlim([x(1),x(end)]);
ylim([y(1),y(end)]);
zlim([z(1),z(end)]);
