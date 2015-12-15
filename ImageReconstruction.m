% Masaharu Ono

% Import greyscale natural image
% img = rgb2gray(imread('640x480_Natural Image2.jpg'));
img = rgb2gray(imread('MNIST2.jpg'));
% display(size(img));

% Take samples of the image
figure(1);
title('Original Image');
clf;
imagesc(img);
colormap('Gray');

% Sample 8x8 patches
image_samples = im2col(img, [8,8], 'distinct');
% display(size(image_samples))

% Compute principle components
A = double(image_samples);
[U, S ,V] = svd(A) ;

% display(size(U(:, 1:2)));
% display(size(S(1:2,1:2)))
% display(size(V(:, 1:2)))

% Display first 64 columns
figure(2);
title('First 64 Columns');
clf;
k = 1;
for i = 1:64
    reshaped = reshape(U(:,i), 8,8);
    subplot(8, 8, k);
    imagesc(reshaped);
    colormap('Gray');
    k = k + 1;
end

% Recontruct images
% A_approx = U(:, 1:R)*S(1:R, 1:R)*V(:, 1:R);
figure(3);
title('R = 2');
clf;
A_approx = U(:, 1:2)*S(1:2, 1:2)*transpose(V(:, 1:2));
reshaped = col2im(A_approx, [8 8], [640 480], 'distinct');
imagesc(reshaped);
colormap('Gray');

figure(4);
title('R = 8');
clf;
A_approx = U(:, 1:8)*S(1:8, 1:8)*transpose(V(:, 1:8));
reshaped = col2im(A_approx, [8 8], [640 480], 'distinct');
imagesc(reshaped);
colormap('Gray');

figure(5);
title('R = 32');
clf;
A_approx = U(:, 1:32)*S(1:32, 1:32)*transpose(V(:, 1:32));
reshaped = col2im(A_approx, [8 8], [640 480], 'distinct');
imagesc(reshaped);
colormap('Gray');
