myFolderdepth = '/home/mahmoud/Downloads/Depth_Estimation_network/data_acquisition/test/depth/';
savingFolderdepth = '/home/mahmoud/Downloads/Depth_Estimation_network/data_acquisition/test/depth1/';

myFolderrgb = '/home/mahmoud/Downloads/Depth_Estimation_network/data_acquisition/test/rgb/';
savingFolderrgb = '/home/mahmoud/Downloads/Depth_Estimation_network/data_acquisition/test/rgb1/';


filePattern = fullfile(myFolderdepth, '*.png');
filePattern2 = fullfile(myFolderrgb, '*.png');

jpegFiles1 = dir(filePattern);
jpegFiles2 = dir(filePattern2);

for k = 1:length(jpegFiles1)
  baseFileName1 = jpegFiles1(k).name;
  baseFileName2 = jpegFiles2(k).name;
  fullFileName1 = fullfile(myFolderdepth, baseFileName1);
  fullFileName2 = fullfile(myFolderrgb, baseFileName2);
  fprintf(1, 'Now reading %s\n', fullFileName1);
  imageArray1 = imread(fullFileName1);
  imageArray2 = imread(fullFileName2);
  
  imageArray3 = flip(imageArray1,1);
  imageArray4 = flip(imageArray2,1);
  % Save them.....
  imwrite(imageArray3,sprintf('/home/mahmoud/Downloads/Depth_Estimation_network/data_acquisition/test/depth1/000100%d.png',k))
  imwrite(imageArray4,sprintf('/home/mahmoud/Downloads/Depth_Estimation_network/data_acquisition/test/rgb1/000100%d.png',k))
  
  imageArray5 = flip(imageArray1,2);
  imageArray6 = flip(imageArray2,2);
  % Save them ......
  imwrite(imageArray5,sprintf('/home/mahmoud/Downloads/Depth_Estimation_network/data_acquisition/test/depth1/000200%d.png',k))
  imwrite(imageArray6,sprintf('/home/mahmoud/Downloads/Depth_Estimation_network/data_acquisition/test/rgb1/000200%d.png',k))
  
  imageArray7 = flip(imageArray5,1);
  imageArray8 = flip(imageArray6,1);
  
  imwrite(imageArray7,sprintf('/home/mahmoud/Downloads/Depth_Estimation_network/data_acquisition/test/depth1/000300%d.png',k))
  imwrite(imageArray8,sprintf('/home/mahmoud/Downloads/Depth_Estimation_network/data_acquisition/test/rgb1/000300%d.png',k))
  
  %a = imageArray(:,:)*10;
  %fullFileName2 = fullfile(savingFolder, baseFileName);
  %imwrite(a,fullFileName2);
  
end