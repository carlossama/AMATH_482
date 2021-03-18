clear all; close all; clc
%% Load Data

path_to_digits = 'D:\AMATH_482\HW_4\train-images.idx3-ubyte';
path_to_labels = 'D:\AMATH_482\HW_4\train-labels.idx1-ubyte';
[IMAGES, LABELS] = mnist_parse(path_to_digits, path_to_labels);
nums = {'zero','one','two','three','four','five','six','seven','eight','nine'};

%% Rearranging data

s = size(IMAGES);
DATA = zeros(s(1)*s(2),s(3));
[LABELS, idx] = sort(LABELS);
for i = 1:s(3)
    image = im2double(IMAGES(:,:,idx(i)));
    image = image(:);
    DATA(:,i) = image;
end  

DATA = DATA - mean(DATA,2);

LABELS_idx = struct();
first = 1;
for i = 0:9
    last = length(find(LABELS == i)) + first - 1;
    LABELS_idx.(nums{i+1}) = [first last];
    first = last + 1;
end


% clearvars IMAGES

%% Train MATLAB models

data_svm = DATA';
mdl = fitcecoc(data_svm, LABELS);
tree = fitctree(data_svm, LABELS);

%% SVD of Data

[U, S, V] = svd(DATA, 'econ');

%% Plot Singular Values

singVals = diag(S);
singVals(singVals < 1e-10) = 0;
figure(14)
plot(singVals,'o');
figure(2)
semilogy(singVals,'o');

%% Plot first nine elements

figure
for k = 1:9
    subplot(3,3,k)
    ut1 = reshape(U(:,k),28,28);
    ut2 = rescale(ut1);
    imshow(ut2)
end


%% Projection

RES.data = S*V';

%% Calculate scatter matrices

features = [2 3 5];
RES.mean = mean(RES.data(features,:),2);

for i = 1:10
    RES.(nums{i}).data = RES.data(features, LABELS_idx.(nums{i})(1):LABELS_idx.(nums{i})(2));
    RES.(nums{i}).mean = mean(RES.(nums{i}).data, 2);
end


Sb = zeros(length(features),1);
for i = 1:10
    Sb = Sb + (RES.(nums{i}).mean - RES.mean)*(RES.(nums{i}).mean - RES.mean)';
end

Sw = zeros(length(features),1);
for i = 1:10
    j = 0;
    while j < length(RES.(nums{i}).data)
        j = j+1;
        Sw = Sw + (RES.(nums{i}).data(:,j) - RES.(nums{i}).mean)*(RES.(nums{i}).data(:,j) - RES.(nums{i}).mean)';
    end
end

%% Plot Projections
close all
figure(3)
hold on
for i = 1:10
    plot3(RES.(nums{i}).data(1,:),RES.(nums{i}).data(2,:),RES.(nums{i}).data(3,:),'o')
    
end
legend(nums);
title('Projection of Each Digit onto Three Principal Components')
xlabel('second mode')
ylabel('third mode')
zlabel('fifth mode')
%% Best Projection Line

[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

%% Project onto w

for i = 1:10
    RES.(nums{i}).proj = w'*RES.(nums{i}).data;
end

%% Plot projections

figure(4)
hold on
for i = 1:10
    plot(RES.(nums{i}).proj, ones(size(RES.(nums{i}).proj))*(i-1),'o');
end

%% pick two and find threshold

NINE = RES.nine.proj;
ONE = RES.one.proj;

if mean(ONE) < mean(NINE)
    warning('uh oh, you didn''t set up your code for this scenario')
end

NINE_sort = sort(NINE);
ONE_sort = sort(ONE);
t1 = length(ONE_sort);
t2 = 1;

while ONE_sort(t1) > NINE_sort(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end

threshold = (ONE_sort(t1) + NINE_sort(t2))/2;

NINE_error = length(find(NINE < threshold));
NINE_errpct = NINE_error / length(NINE);
ONE_error = length(find(ONE > threshold));
ONE_errpct = ONE_error / length(ONE);


%% Plot Seperation

close all
figure(5)
subplot(1,2,1)
histogram(ONE_sort, 60); hold on, plot([threshold threshold], [0 400], 'r')
title('One''s')
subplot(1,2,2)
histogram(NINE_sort, 60); hold on, plot([threshold threshold], [0 400], 'r')
title('Nine''s')

%% Three Digits

FOUR = RES.four.proj;
FOUR_sort = sort(FOUR);

if mean(FOUR) < mean(NINE)
    warning('uh oh, you didn''t set up your code for this scenario')
end

threshold_1_9 = threshold;

t1 = length(ONE_sort);
t2 = 1;
while ONE_sort(t1) > FOUR_sort(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold_1_4 = (ONE_sort(t1) + FOUR_sort(t2))/2;

t1 = length(NINE_sort);
t2 = 1;
while NINE_sort(t1) > FOUR_sort(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold_9_4 = (NINE_sort(t1) + FOUR_sort(t2))/2;

NINE_errpct_1 = NINE_errpct;
ONE_errpct_9 = ONE_errpct;
NINE_errpct_4 = length(find(NINE > threshold_9_4)) / length(NINE);
ONE_errpct_4 = length(find(ONE > threshold_1_4)) / length(ONE);

FOUR_err = length(find(FOUR < threshold_1_4));
FOUR_errpct_1 = FOUR_err / length(FOUR);
FOUR_err = length(find(FOUR < threshold_9_4));
FOUR_errpct_9 = FOUR_err / length(FOUR);



%% Load test data

path_to_digits = 'D:\AMATH_482\HW_4\t10k-images.idx3-ubyte';
path_to_labels = 'D:\AMATH_482\HW_4\t10k-labels.idx1-ubyte';
[IMAGES, LABELS] = mnist_parse(path_to_digits, path_to_labels);


s = size(IMAGES);
DATA = zeros(s(1)*s(2),s(3));
[LABELS, idx] = sort(LABELS);
for i = 1:s(3)
    image = im2double(IMAGES(:,:,idx(i)));
    image = image(:);
    DATA(:,i) = image;
end  

DATA = DATA - mean(DATA,2);

LABELS_idx = struct();
first = 1;
for i = 0:9
    last = length(find(LABELS == i)) + first - 1;
    LABELS_idx.(nums{i+1}) = [first last];
    first = last + 1;
end

%% Projection

TEST.data = U'*DATA;

for i = 1:10
    TEST.(nums{i}).data = TEST.data(features, LABELS_idx.(nums{i})(1):LABELS_idx.(nums{i})(2));
    TEST.(nums{i}).mean = mean(TEST.(nums{i}).data, 2);
end
for i = 1:10
    TEST.(nums{i}).proj = w'*TEST.(nums{i}).data;
end

treeRes = predict(tree, DATA');

svmRes = predict(mdl, DATA');

%% Classify

prediction_LDA = zeros(size(LABELS));
types = {'LDA','tree','SVM'};

for k = 1:3
    for i = 1:10
        for j = 1:10
            TEST.(nums{i}).(types{k}).(nums{j}) = 0;
        end
    end
end

NINEt = TEST.nine.proj;
ONEt = TEST.one.proj;
FOURt = TEST.four.proj;
NINE_error = length(find(NINEt < threshold));
NINEt_errpct = NINE_error / length(NINEt);
ONE_error = length(find(ONEt > threshold));
ONEt_errpct = ONE_error / length(ONEt);

NINEt_errpct_1 = NINEt_errpct;
ONEt_errpct_9 = ONEt_errpct;
NINEt_errpct_4 = length(find(NINEt > threshold_9_4)) / length(NINEt);
ONEt_errpct_4 = length(find(ONEt > threshold_1_4)) / length(ONEt);

FOURt_err = length(find(FOURt < threshold_1_4));
FOURt_errpct_1 = FOURt_err / length(FOURt);
FOURt_err = length(find(FOURt < threshold_9_4));
FOURt_errpct_9 = FOURt_err / length(FOURt);


for i = 1:length(TEST.proj)
    TEST.(nums{LABELS(i)+1}).tree.(nums{treeRes(i) + 1}) = TEST.(nums{LABELS(i)+1}).tree.(nums{treeRes(i) + 1}) + 1;
end

for i = 1:length(TEST.proj)
    TEST.(nums{LABELS(i)+1}).SVM.(nums{svmRes(i) + 1}) = TEST.(nums{LABELS(i)+1}).SVM.(nums{svmRes(i) + 1}) + 1;
end

%%

close all
figure(3)
hold on
for i = 1:10
    plot3(TEST.(nums{i}).data(2,:),TEST.(nums{i}).data(1,:),TEST.(nums{i}).data(3,:),'o')
    pause(1)
end
%%
for i = 1:10
    TEST.(nums{i}).proj = w'*TEST.(nums{i}).data;
end
%%

figure(11)
hold on
for i = 1:10
    plot(TEST.(nums{i}).proj, ones(size(TEST.(nums{i}).proj))*(i-1),'o');
end
