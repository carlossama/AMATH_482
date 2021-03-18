clear all; close all; clc

%% Load data for instance

tests = [1, 2, 3, 4];
camList = [1 2 3];  


crop.cam1 = [125, 250, 460, 450]; %ijij
crop.cam2 = [75, 150, 410, 400];
crop.cam3 = [150, 200, 500, 500];
show = 1;


instance = 1;

for camera = 1:length(camList) % length(camList)
    CAMS.(strcat('cam',num2str(camList(camera)))) = load([strcat('cam',num2str(camList(camera))),'_',num2str(tests(instance)),'.mat']);
    CAMS.(strcat('cam',num2str(camList(camera)))).size = size(CAMS.(strcat('cam',num2str(camList(camera)))).(strcat('vidFrames',num2str(camList(camera)),'_',num2str(instance))));
    dataSize = CAMS.(strcat('cam',num2str(camList(camera)))).size;
    x = linspace(0, dataSize(1), dataSize(1)+1);
    y = linspace(0, dataSize(2), dataSize(2)+1);
    x = x(1:end-1);
    y = y(1:end-1);
    kx = (2*pi/(2*dataSize(1)))*[0:(dataSize(1)/2-1) -dataSize(1)/2:-1];    
    ky = (2*pi/(2*dataSize(2)))*[0:(dataSize(2)/2-1) -dataSize(2)/2:-1];    
    [X,Y] = meshgrid(x,y); 
    [Kx,Ky] = meshgrid(kx,ky); 
    X = X'; Y = Y';
    Kx = Kx'; Ky = Ky';
    t = dataSize(4);
    points = zeros(t,2);
    vid = CAMS.(strcat('cam',num2str(camList(camera)))).(strcat('vidFrames',num2str(camList(camera)),'_',num2str(instance)));
    ut_avg = zeros(dataSize(1), dataSize(2));
    cropFilt = zeros(size(ut_avg));
    cropVals = crop.(strcat('cam',num2str(camList(camera))));
    for i = cropVals(1):cropVals(3)
        for j = cropVals(2):cropVals(4)
            cropFilt(i,j) = 1;
        end
    end
    
    for i = 1:t
        u = double(rgb2gray(vid(:,:,:,i)));
        u = u.*cropFilt;
        ut = fftn(u);
        ut_avg = ut_avg + ut;
        
    end
    %ut_avg = ut_avg/t;
    [idx1, idx2] = ind2sub([length(x),length(y)], find(ut_avg == max(ut_avg, [], 'all')));
    km = [Kx(idx1, idx2), Ky(idx1, idx2)];
    a = 100;
    filter = exp(-a*((Kx - km(1)).^2+(Ky - km(2)).^2));
    for i = 1:t
        u = double(rgb2gray(vid(:,:,:,i))).*cropFilt;
        ut = fftn(u);
        utc = ut;
        utf = utc.*filter;
        frameFilt = real(ifftn(utf));
        [idx1, idx2] = ind2sub([length(x),length(y)], find(abs(frameFilt) == max(abs(frameFilt), [], 'all')));
        points(i,:) = [x(idx1), y(idx2)];
        if show == 1
            pcolor(flipud(abs(frameFilt)/max(abs(frameFilt), [], 'all')))
            shading interp, colormap(gray); 
            grid on
            hold on
            plot(points(i,2), -(points(i,1)-length(x)), 'm.-', 'MarkerSize', 20);
            hold off
            drawnow
        end
    end

    k = [0:(t/2-1) -t/2:-1];
    a = 0.003; 
    filter = exp(-a*(k).^2);
    filter = filter';

    x = real(ifft(fft(points(1:length(k),2)).*filter));
    y = real(ifft(fft(points(1:length(k),1)).*filter));
    CAMS.(strcat('cam',num2str(camList(camera)))).x = x;
    CAMS.(strcat('cam',num2str(camList(camera)))).y = y;
    CAMS.(strcat('cam',num2str(camList(camera)))).points = points;


    
end
   
%%
minim = 99999999999;
for i = 1:length(camList)
    if length(CAMS.(strcat('cam',num2str(camList(i)))).x) < minim
        minim = length(CAMS.(strcat('cam',num2str(camList(i)))).x);
    end
end

data = zeros(minim, 6);
       
for i = 0:length(camList)-1
    xdata = CAMS.(strcat('cam',num2str(camList(i+1)))).x;
    ydata = CAMS.(strcat('cam',num2str(camList(i+1)))).y;
    xdata = xdata - mean(xdata);
    ydata = ydata - mean(ydata);
    data(:,(i*2 + 1)) = xdata(1:minim);
    data(:,(i*2 + 2)) = ydata(1:minim);
end
data = data';
A = data * 1/(sqrt(length(data) - 1)) ;
[u, s, v] = svd(A*A');
y = u' * data;

%%
close all
figure(1)
subplot(3,1,1);
hold on
plot(data(1,:));
plot(data(3,:));
plot(data(5,:));
title('measured x displacement data');
legend('camera 1','camera 2','camera 3')

subplot(3,1,2);
hold on
plot(data(2,:));
plot(data(4,:));
plot(data(6,:));
title('measured y displacement data');
legend('camera 1','camera 2','camera 3')


subplot(3,1,3);
hold on
plot(y(1,:));
plot(y(2,:));
plot(y(3,:));
title('principal component interpretation')
legend('comp 1','comp 2','comp 3');


figure(2)
s(s<10e-13) = 10e-13;
pvals = s(logical(eye(6)));
plot(pvals, '*', 'MarkerSize',10,'Color','r');
title('variance components');











