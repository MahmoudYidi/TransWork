

world_cam = 'path to camera poses';
world_target = 'path to target poses';

Nos_image = 835; % Number of Images

posevecs_oc = readmatrix(world_cam);
posevecs_ot = readmatrix(world_target);
posevecs_oc = iMatPts2CellPts(posevecs_oc, 7);
posevecs_ot = iMatPts2CellPts(posevecs_ot, 7);
A = zeros(Nos_image,7); 


for i = 1:Nos_image
    posevec_oc = posevecs_oc{1};
    posevec_ot = posevecs_ot{1};
    
   
    T_oc = optitrack_build_posemat(posevec_oc(5:7),posevec_oc(1:4));
    T_ot = optitrack_build_posemat(posevec_ot(5:7),posevec_ot(1:4));
    T_ct = T_ot \ T_oc
    %T_ct = inv(T_ct) %Uncomment if you want to compute the inverse
    q = so3_to_su2(T_ct(1:3,1:3));
    t = T_ct(1:3,4);
    posevec = [q' t']
    norm(posevec(1:4))
    A(i,:) = posevec;
    
end
writematrix(A,'final_relative_pose_raw.csv')