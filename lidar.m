bagInfo = rosbag('info','2019-03-12-19-53-04.bag')
bag = rosbag('2019-03-12-19-53-04.bag');
rosbag info '2019-03-12-19-53-04.bag'
bagselect2 = select(bag,'Time',...
    [bag.StartTime bag.StartTime + 0.1],'Topic','/cloud');
msgs = readMessages(bag);
ptCloud= msgs{1};

xyz = readXYZ(ptCloud);
ptcloud = pointCloud(xyz)
pcshow(ptcloud);
view([180 90]);

% scatter3(ptCloud)
% model = pcfitplane(ptCloudIn,maxDistance)
% B=reshape(xyz,920,24,3)
% surf(B(:,:,3))
% model = pcfitplane(ptCloudIn,maxDistance)
% ptCloud = pcdenoise(ptCloud)
% fieldnames = readAllFieldNames(ptCloud)

x=[];y=[];z=[];
x = readField(ptCloud,'x');
x=reshape(x,920,24);
x=x(506:605,:);

y = readField(ptCloud,'y');
y=reshape(y,920,24);
y=y(506:605,:);

z = readField(ptCloud,'z');
z=reshape(z,920,24);
z=z(506:605,:);

plot(x)
hold on 
plot(y)
plot(z)

x_=[];
for i=1:24
   x_(:,i) = linspace(x(1,i),x(end,i),100).';
end 
    
%%
out1 = x - repmat(x(:,1), [1, 24]);
figure
NC=10;
newmap = (gray(NC));    
colormap(newmap);      
im = imagesc((out1));
h = colorbar(); cl = caxis;
shading interp;       

%%
out2 = x - repmat(x(1,:), [100,1]);
figure
NC=10;
newmap = (gray(NC));    
colormap(newmap);      
im = imagesc(out2);
h = colorbar(); cl = caxis;


%%
out0 = x_ - x;
figure
NC=10;
newmap = (gray(NC));    
colormap(newmap);      
im = imagesc((out0));
h = colorbar(); cl = caxis;
shading interp;   
