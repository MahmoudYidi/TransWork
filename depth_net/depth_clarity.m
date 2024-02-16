myFolderdepth = '/home/mahmoud/Downloads/testing_depth/depth_try';
savingFolderdepth = '/home/mahmoud/Downloads/testing_depth/depth1/';

filePattern = fullfile(myFolderdepth, '*.png');


pngFiles = dir(filePattern);

for k = 1:length(pngFiles)
  baseFileName1 = pngFiles(k).name;
  
  fullFileName1 = fullfile(myFolderdepth, baseFileName1);
  
  fprintf(1, 'Now reading %s\n', fullFileName1);
  imageArray1 = imread(fullFileName1);
  
  imageArray1 = imageArray1 .* 10;
  
  
  % Save them.....
  imwrite(imageArray1,sprintf('/home/mahmoud/Downloads/testing_depth/depth1/mage_0000%d.png',k+4));
  
end