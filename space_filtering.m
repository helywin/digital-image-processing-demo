close all;
clearvars;

origin = imread('assets/gakki.jpg');
imshow(origin);

gray = rgb2gray(origin);
gaussian_noise = imnoise(gray, 'gaussian');
figure;
imshow(gaussian_noise);

f = fft2(gaussian_noise);
f = fftshift(f);
figure;
imshow(log(abs(f)+1), []);