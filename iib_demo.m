clc; clearvars; close all; addpath('.\mexopencv\')

% read images
image_ref                               = imread('ref_img.png');
image_similar                           = imread('similar_img.png');
image_dissimilar                        = imread('dissimilar_img.png');

% load keypoints
kp                                      = load("kp.mat").kp;

% extract IIB descriptor
[kp_ref, desc_ref]                      = gen_iib(kp, rgb2gray(image_ref),          4, 1);
[kp_similar, desc_similar]              = gen_iib(kp, rgb2gray(image_similar),      4, 1);
[kp_dissimilar, desc_dissimilar]        = gen_iib(kp, rgb2gray(image_dissimilar),   4, 1);

% obtain keypoint locations
kp_ref_loc                              = vertcat(kp_ref.pt);
kp_similar_loc                          = vertcat(kp_similar.pt);
kp_dissimilar_loc                       = vertcat(kp_dissimilar.pt);

% match descriptors
matcher                                 = DescriptorMatcher('BFMatcher', 'NormType', 'Hamming', 'CrossCheck', true);

matches                                 = matcher.match(desc_ref, desc_similar);
index_pairs                             = [[matches.queryIdx]', [matches.trainIdx]'];
index_pairs_similar                     = index_pairs + 1;

matches                                 = matcher.match(desc_ref, desc_dissimilar);
index_pairs                             = [[matches.queryIdx]', [matches.trainIdx]'];
index_pairs_dissimilar                  = index_pairs + 1;

% validate matches
diff                                    = kp_ref_loc(index_pairs_similar(:, 1), :) - kp_similar_loc(index_pairs_similar(:, 2), :);
dis                                     = sqrt(sum(diff.^2, 2));
index_pairs_similar                     = index_pairs_similar(dis < 0.1, :);

diff                                    = kp_ref_loc(index_pairs_dissimilar(:, 1), :) - kp_dissimilar_loc(index_pairs_dissimilar(:, 2), :);
dis                                     = sqrt(sum(diff.^2, 2));
index_pairs_dissimilar                  = index_pairs_dissimilar(dis < 0.1, :);

% display matches
fig = figure;
subplot(211)
showMatchedFeatures(image_ref, image_similar, kp_ref_loc(index_pairs_similar(1:10:end, 1), :), kp_similar_loc(index_pairs_similar(1:10:end, 2), :), "montage");
title(['Matched feature points in two images with similar illuminations: ', num2str(size(index_pairs_similar(1:10:end, :), 1))]);
subplot(212)
showMatchedFeatures(image_ref, image_dissimilar, kp_ref_loc(index_pairs_dissimilar(1:10:end, 1), :), kp_dissimilar_loc(index_pairs_dissimilar(1:10:end, 2), :), "montage");
title(['Matched feature points in two images with dissimilar illuminations: ', num2str(size(index_pairs_dissimilar(1:10:end, :), 1))]);
