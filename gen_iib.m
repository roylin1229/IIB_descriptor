% Please note that the current implementation does not consider the primary orientation information of feature points.

% Input:
% kp            - OpenCV formatted feature points
% img_gray      - Input image in grayscale
% g             - Spatial granularity
% radiu_scale   - Scaling factor for constructing the ROS of descriptors

% Output:
% kp            - Valid OpenCV formatted feature points
% descs         - Valid feature point descriptors
% alls_inliers  - Indices of valid points

function [kp, descs, alls_inliers] = gen_iib(kp, img_gray, g, radiu_scale)

[row, col]              = size(img_gray);

kp_size                 = [kp.size]' ./ 2;
kp_loc                  = round(vertcat(kp.pt))+1;                              
radius                  = round(radiu_scale * kp_size);                         
temp_rois               = round([kp_loc-radius 2*radius 2*radius]);             
alls_inliers            = temp_rois(:, 1) > 1 & temp_rois(:, 1)+temp_rois(:, 3) < col & ...
                          temp_rois(:, 2) > 1 & temp_rois(:, 2)+temp_rois(:, 4) < row;

kp                      = kp(alls_inliers);
kp_loc                  = kp_loc(alls_inliers, :);
radius                  = radius(alls_inliers);

if isempty(kp)
    descs               = [];
    return;
end

[radius_unique, ~, radiu_idx]   = unique(radius);

for i = 1:length(radius_unique)
    param.rois{i}          = get_rois([radius_unique(i)+1 radius_unique(i)+1], radius_unique(i), g);
    param.rois{i}          = cell2mat(param.rois{i}');
    param.rois{i}          = round(param.rois{i});                                
    [param.rois_indices{i}, param.cp_linear_indices{i}] = get_linear_indices([row, col], param.rois{i}, radius_unique(i));
end

param.roi_num       = arrayfun(@(x) 4^x, 1:g);

param.desc_dim      = sum(param.roi_num ./ 4);
param.desc_dims     = param.roi_num ./ 4;

param.sobel_x       = [-1  0  1;  -2 0 2; -1 0 1];
param.sobel_y       = [-1 -2 -1;   0 0 0;  1 2 1];

img_gx              = imfilter(single(img_gray), param.sobel_x, 'replicate');
img_gy              = imfilter(single(img_gray), param.sobel_y, 'replicate');
img_gx_abs          = abs(img_gx);
img_gy_abs          = abs(img_gy);

img_go              = atan2d(-img_gy, img_gx)+180;

num_img             = uint8(ones(size(img_gray)));
i_img_num           = uint64(integralImage(num_img));
i_img_num           = i_img_num(:);

i_img_gx_abs        = uint64(integralImage(img_gx_abs));
i_img_gx_abs        = i_img_gx_abs(:);

i_img_gy_abs        = uint64(integralImage(img_gy_abs));
i_img_gy_abs        = i_img_gy_abs(:);

i_img_go            = integralImage(img_go);
i_img_go            = i_img_go(:);

i_img_gray          = uint64(integralImage(img_gray));
i_img_gray          = i_img_gray(:);

kp_loc_linear_inds   = sub2ind([row+1, col+1], kp_loc(:, 2)+1, kp_loc(:, 1)+1);

temp_rois_indices    = cell(1, length(kp));
for i = 1:length(kp)
    temp_rois_indices{i}    = param.rois_indices{radiu_idx(i)} + kp_loc_linear_inds(i) - param.cp_linear_indices{radiu_idx(i)};
end
temp_rois_indices = cell2mat(temp_rois_indices');
temp_rois_indices = uint64(temp_rois_indices);

desc_gx_abs         = my_desc(i_img_gx_abs,     i_img_num, temp_rois_indices);
desc_gy_abs         = my_desc(i_img_gy_abs,     i_img_num, temp_rois_indices);
desc_go             = my_desc(i_img_go,         i_img_num, temp_rois_indices);
desc_gray           = my_desc(i_img_gray,       i_img_num, temp_rois_indices);

desc_gx_abs         = reshape(desc_gx_abs,      [param.desc_dim, length(kp)]);
desc_gy_abs         = reshape(desc_gy_abs,      [param.desc_dim, length(kp)]);
desc_go             = reshape(desc_go,          [param.desc_dim, length(kp)]);
desc_gray           = reshape(desc_gray,        [param.desc_dim, length(kp)]);

desc_gx_abs         = mat2cell(desc_gx_abs,     param.desc_dims, length(kp));
desc_gy_abs         = mat2cell(desc_gy_abs,     param.desc_dims, length(kp));
desc_go             = mat2cell(desc_go,     param.desc_dims, length(kp));
desc_gray           = mat2cell(desc_gray,     param.desc_dims, length(kp));

descs               = cell(g, 1);
for j = 1:g
    descs{j}        = [desc_gx_abs{j}*16+desc_gy_abs{j}; desc_go{j}*16+desc_gray{j}];
end

descs               = cell2mat(descs)';

if length(kp) ~= size(descs, 1) 
    error('wrong ')
end

end


%%
function rois = get_rois(point, radiu, g)

rois    = cell(1, g);
for i   = 1:g
    rois{i} = zeros(4^i, 4);
end

init_roi        = [point-radiu point+radiu];
temp_rois       = init_roi;

for i = 1:g
    next_rois = zeros(size(temp_rois, 1)*4, 4);
    for j = 1:size(temp_rois, 1)
        sub_rois = split_box(temp_rois(j, :));
        next_rois((j-1)*4+1:j*4, :) = sub_rois;
    end
    
    temp_rois   = next_rois;
    rois{i}     = temp_rois;
end

end


%%
function [sub_rois] = split_box(roi)

c_m = (roi(1) + roi(3)) / 2;
r_m = (roi(2) + roi(4)) / 2;
sub_rois = [[roi(1) roi(2) c_m r_m]; [roi(1) r_m c_m roi(4)]; [c_m roi(2) roi(3) r_m]; [c_m r_m roi(3) roi(4)]];

end


%%
function [descs] = my_desc(i_img, i_img_num, rois_indices)

temp_values_raw             = i_img(rois_indices(:, 4)) + i_img(rois_indices(:, 1)) - i_img(rois_indices(:, 2)) - i_img(rois_indices(:, 3));
temp_nums                   = i_img_num(rois_indices(:, 4)) + i_img_num(rois_indices(:, 1)) - i_img_num(rois_indices(:, 2)) - i_img_num(rois_indices(:, 3));
temp_values_raw             = double(temp_values_raw) ./ double(temp_nums);

temp_values_raw             = reshape(temp_values_raw, [4 size(temp_values_raw, 1)/4]);

[descs]                     = bin_mean(temp_values_raw);

end


%%
function [final_desc] = bin_mean(temp_values_intra)

mean_values_intra               = mean(temp_values_intra);
descs_intra                     = temp_values_intra > mean_values_intra;
descs_intra                     = descs_intra';

final_desc                      = 8.*descs_intra(:, 1) + 4.*descs_intra(:, 2) + 2.*descs_intra(:, 3) + descs_intra(:, 4);
final_desc                      = uint8(final_desc);

end


%%
function [linear_indices, cp_linear_indices] = get_linear_indices(img_size, rois, radiu)

rois                      = rois + 1;

inds_2_2                  = sub2ind([img_size(1)+1, img_size(2)+1], rois(:, 4),     rois(:, 3));
inds_1_1                  = sub2ind([img_size(1)+1, img_size(2)+1], rois(:, 2)-1,   rois(:, 1)-1);
inds_1_2                  = sub2ind([img_size(1)+1, img_size(2)+1], rois(:, 4),     rois(:, 1)-1);
inds_2_1                  = sub2ind([img_size(1)+1, img_size(2)+1], rois(:, 2)-1,   rois(:, 3));

linear_indices            = [inds_1_1 inds_1_2 inds_2_1 inds_2_2];

cp_linear_indices         = sub2ind([img_size(1)+1, img_size(2)+1], radiu+2, radiu+2);

end
