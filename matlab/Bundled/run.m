% Bundled Camera Path Video Stabilization
% Written by Tan SU
% contact: sutank922@gmail.com

clear all;
addpath('mesh');
addpath('RANSAC');
addpath('mex');

%% Parametres
% -------INPUT-------
classname={'Regular','Parallax','Crowd','Bundled/Crowd','para_1_20_10_40_20_error_2147483647/Regular','para_1_20_10_40_40_iter_1/Regular','Crowd','Parallax'};
for n=2:3
    caseFile=strcat('E:/data/BUNDLED2/images/',classname{1,n},'/');
    num=length(dir(strcat(caseFile,'*.avi')));
    doc=dir(strcat(caseFile,'*.avi'));
    casename=cell(1,num);
    for i=1:length(casename)
        casename{1,i}=doc(i).name;%num2str(i-1);
    end
    
    for i=5:length(casename)
        inputDir = strcat(caseFile,casename{1,i},'/');
        outputDir = strcat('E:/data/BUNDLED2/results_images/',classname{1,n},'/',casename{1,i},'/');
        mkdir(outputDir);
        nFrames =  length(dir(strcat(inputDir,'/*.png')));
        % -------TRACK-------
        TracksPerFrame = 512;           % number of trajectories in a frame, 200 - 2000 is OK
        % -------STABLE------
        MeshSize = 16;                  % The mesh size of bundled camera path, 6 - 12 is OK
        Smoothness = 3;                 % Adjust how stable the output is, 0.5 - 3 is OK
        Span = 60;                      % Omega_t the window span of smoothing camera path, usually set it equal to framerate
        Cropping = 1;                   % adjust how similar the result to the original video, usually set to 1
        Rigidity = 2;                   % adjust the rigidity of the output mesh, consider set it larger if distortion is too significant, [1 - 4]
        iteration = 20;                 % number of iterations when optimizing the camera path [10 - 20]
        % -------OUTPUT------
        OutputPadding = 200;            % the padding around the video, should be large enough. 

        %% Track by KLT
        tic;
        track = GetTracks(inputDir, MeshSize, TracksPerFrame, nFrames); 
        toc;

        %% Compute original camera path (by As-similar-as-possible Warping)
        tic;
        path = getPath(MeshSize, track);    
        toc;

        %% Optimize the paths
        tic;
        bundled = Bundled(inputDir, path, Span, Smoothness, Cropping, Rigidity);
        bundled.optPath(iteration);
        toc;

        %% Render the stabilzied frames
        tic;
        bundled.render(outputDir, OutputPadding);
        toc;
    end
end
