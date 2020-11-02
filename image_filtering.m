close all;
clearvars;
origin = imread('assets/dark_character.jpg');
gray = rgb2gray(origin);
gray = im2double(gray);
imshow(gray);
f = fft2(gray);
f = fftshift(f);
[angle, radius] = cart2pol(real(f), imag(f));
magnitude_log = log(radius + 1);
figure;
imshow(magnitude_log,[]);

imgsize = size(f);

%pol2cart


center = fliplr(imgsize./2);

d1 = 50;
d2 = 50;
r = 200;
n = 2.5;
butterworth_low = zeros(imgsize);
butterworth_high = zeros(imgsize);
laplace = zeros(imgsize);
for i = 1 : imgsize(1)
    for j = 1 : imgsize(2)
        butterworth_low(i,j)=1/(1+power(norm([j,i] - center)/d1, 2 * n));
        butterworth_high(i,j)=1/(1+power(d2/norm([j,i] - center), 2 * n));
        laplace(i,j)=-4 * pi * power(norm([j,i] - center), 2)/r;
    end
end
filtered_radius = radius.*butterworth_low;
[x, y] = pol2cart(angle, filtered_radius);
f1 = x + 1i*y;
after1 = ifft2(f1);
figure
imshow(abs(after1), []);

filtered_radius = radius.*butterworth_high;
[x, y] = pol2cart(angle, filtered_radius);
f1 = x + 1i*y;
after2 = ifft2(f1);
figure
imshow(abs(after2), []);
% imshow(abs(after2)+gray, []);

filtered_radius = radius.*laplace;
[x, y] = pol2cart(angle, filtered_radius);
f1 = x + 1i*y;
after3 = ifft2(f1);
figure
abs3 = abs(after3);
imshow(abs3, []);
% imshow(abs3./max(abs3, [], 'all') + gray, []);




