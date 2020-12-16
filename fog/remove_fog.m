close all;
clearvars;

fog = imread('tian1.jpg');

subplot(2,2,1), imshow(fog(:,:,1))
subplot(2,2,2), imshow(fog(:,:,2))
subplot(2,2,3), imshow(fog(:,:,3))
subplot(2,2,4), imshow(fog)
% find darkness channel
min_channel = 1;

min_r = min(fog(:,:,1),[],'all');
min_g = min(fog(:,:,2),[],'all');
min_b = min(fog(:,:,3),[],'all');

mean_r = mean(fog(:,:,1),'all');
mean_g = mean(fog(:,:,2),'all');
mean_b = mean(fog(:,:,3),'all');
min_all = min_r;
mean_all = mean_r;
if mean_g < mean_all
    min_channel = 2;
    min_all = min_g;
    mean_all = mean_g;
end

if mean_b < mean_all
    min_channel = 3;
end
darkness = fog(:,:,min_channel);
darkness = double(darkness);
% light

array_size = size(darkness);
pixel_count = array_size(1,1) * array_size(1,2);
reshaped_array = reshape(darkness, [1, pixel_count]);
sorted_array = sort(reshaped_array, 'descend');
sample_count = round(pixel_count * 0.001);
average_gray = sum(sorted_array(1, 1:sample_count))./sample_count;
% average_gray = 200;

window_width = 3;
window_height = window_width;
half_window_width = floor(window_width/2);
half_window_height = floor(window_height/2);

after = fog;
after = double(after);
tN = zeros(array_size, 'double');

for r=1:array_size(1,1)
    for c=1:array_size(1,2)
        y_start = max(r - half_window_height, 1);
        y_end = min(r + half_window_height, array_size(1,1));
        x_start = max(c - half_window_width, 1);
        x_end = min(c + half_window_height, array_size(1,2));
        tN(r,c) = 1 - min(darkness(y_start:y_end, x_start:x_end), [], 'all')./average_gray;
        after(r, c, 1:3) = floor(average_gray - (average_gray - after(r, c, 1:3))./tN(r,c));
    end
end

after = uint8(after);
figure;
subplot(2,2,1), imshow(after(:,:,1))
subplot(2,2,2), imshow(after(:,:,2))
subplot(2,2,3), imshow(after(:,:,3))
subplot(2,2,4), imshow(after)
imwrite(after, 'after.jpg');