clc;
load digitStruct.mat

%LOAD TRAINING IMAGES
train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');

[size_image train_samples]=size(train_images);

mean=zeros(size_image,1);
A=zeros(size_image, train_samples);
top=20; %HOW MANY TOP EIGEN VALUES SHOULD WE INCLUDE
lambda_top=zeros(top,1);
U_top=zeros(size_image,top); %STORES THE TOP EIGEN VECTORS(EIGEN FACES)
omega=zeros(top,train_samples);

%CALCULATE MEAN
for i=1:train_samples
    mean=mean+train_images(:,i);
end
mean=mean/train_samples;

for i=1:train_samples
    A(:,i)= train_images(:,i)-mean;
end

C = A*A';
[U,lambda] = eig(C,'vector');

%TO TAKE TOP EIGEN VALUES WE SORT
[B, I]=sort(lambda,'descend');

for i=1:top
    lambda_top(i)=lambda(I(i));
    U_top(:,i)=(U(:,I(i)));
end

%CALCULATE WEIGHTS FOR EVERY TRAINING SAMPLE
for i=1:train_samples
    for j=1:top
        omega(j,i)= U_top(:,j)'* train_images(:,i);        
    end
end

%DISPLAY TOP EIGEN VECTORS
display_network(U_top(:,1:10));


%LOAD MNIST TESTING DATA
test_images = loadMNISTImages('train-images.idx3-ubyte');
test_labels = loadMNISTLabels('train-labels.idx1-ubyte');
test_samples=500;
mean_test=zeros(size_image, 1);
A_test=zeros(size_image, test_samples);

for i=1:test_samples
    mean_test=mean_test+test_images(:,i);
end
mean_test=mean_test/test_samples;

for i=1:test_samples
    A_test(:,i)= test_images(:,i)-mean_test;
end
K=5; %k-FOR K NEAREST NEIGHBORS

%CALCULATE WEIGHTS FOR TESTING SAMPLES
for i=1:test_samples
    count_dig = zeros(10,1);
    for j=1:top
        omega_test(j,i)= U_top(:,j)'* test_images(:,i);                                        
    end
end

%IMPLEMENTING K-NN
accuracy=0;
pred_label=zeros(test_samples,1);
for i=1:test_samples
    a=omega_test(:,i);
    b=repmat(omega_test(:,i), 1,train_samples);
    difference= b-omega;
    difference=difference.^2;
    error = sum(difference,1);
    [sort_error error_ind]=sort(error, 'ascend');
    for j=1:K
        k_labels(j)=train_labels(error_ind(j)); %SELECT NEAREST NEIGHBORS
    end
    pred_label(i) = mode(k_labels);%SELECT THE MAXIMUM VALUE
    if( pred_label(i)==test_labels(i))
        accuracy=accuracy+1;
        correct_sample=i;%STORE ONE VALUE TO PLOT A CORRECT SAMPLE
    else
        incorrect_sample=i;%STORE ONE VALUE TO PLOT AN INCORRECT SAMPLE
    end
end


%DISPLAY IMAGES
correct_dig = reshape(test_images(:,correct_sample), [28 28]);
incorrect_dig = reshape(test_images(:,incorrect_sample), [28 28]);
figure;
imshow(correct_dig);
title('correct classified digit');
figure;
imshow(incorrect_dig);
title('incorrect classified digit');

%CALCULATE ACCURACY
error_MNIST = ((test_samples - accuracy)/test_samples)*100;
accu_MNIST = ((accuracy)/test_samples)*100;



%%%%%%%%%%%SVHN%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_samples_svhn=300; 
count=0;
correct_class = 0;
correct_house_num = 0;
path='D:\E\EECS 504\hw3_final\hw3p2\test\'

%FOR ALL TEST SAMPLES
for i=1:test_samples_svhn
    name= digitStruct(i).name;
    bbox= digitStruct(i).bbox;
    [~,len]=size(bbox);
    test_image_svhn=im2double(rgb2gray((imread(strcat(path,name)))));
    %BINARIZE THE TESTING IMAGE
    for p = 1:size(test_image_svhn,1)
        for q = 1:size(test_image_svhn,2)
            if test_image_svhn(p,q)> 0.5
                test_image_svhn(p,q) = 1;
            else
                test_image_svhn(p,q) = 0;
            end
        end
    end 
    num_house=0;
    % FOR EVERY DIGIT IN THE HOUSE NUMBER
    for j=1:len
        [height, width] = size(test_image_svhn);
        xmin = max(digitStruct(i).bbox(j).top+1,1);
        xmax = min(digitStruct(i).bbox(j).top+digitStruct(i).bbox(j).height, height);
        ymin = max(digitStruct(i).bbox(j).left+1,1);
        ymax = min(digitStruct(i).bbox(j).left+digitStruct(i).bbox(j).width, width);
        test_image_resize = imresize(double(test_image_svhn(xmin:xmax, ymin:ymax, :)),[28 28]);
        count=count+1;
        count_dig = zeros(10,1);
         for m=1:top
             omega_test_svhn(m)= U_top(:,m)'* test_image_resize(:);
         end
         % CALCULATE ERROR WITH EVERY TRAINING SAMPLE
         error = sqrt(sum((omega - omega_test_svhn').^2));
        % CHOOSE K-NEAREST NEIGHBORS
         [~,error_ind] = sort(error,'ascend');
        new_pos = error_ind(1:K);
    
        for m = 1:K
            label = train_labels(new_pos(m));
            count_dig(label+1) = count_dig(label+1)+1;
        end
    
        [max_val,pred_label_svhn] = max(count_dig);
        pred_label_svhn = pred_label_svhn - 1;
        
        if pred_label_svhn == digitStruct(i).bbox(j).label
            correct_class = correct_class + 1;
            num_house = num_house + 1;
        end
    end
    %CHECK FOR THE CORRECT HOUSE NUMBER
    if num_house == length(digitStruct(i).bbox)
        correct_house_num = correct_house_num + 1;
        correct_sample=i; %STORE VALUE TO DISPLAY CORRECT SAMPLE
    else 
        incorrect_sample=i; %STORE VALUE TO DISPLAY INCORRECT SAMPLE
    end
end
% CALCULATE ERROR
error_SVHN = (1 - correct_class/count)*100;

%DISPLAY IMAGES
figure;
imshow((imread(strcat(path,digitStruct(correct_sample).name))));
title('correctly classified house number');
figure;
imshow((imread(strcat(path,digitStruct(incorrect_sample).name))));
title('incorrectly classified house number');

