%%% perform linear mixed effects on vergence data

data_struct = load('data/vergence_distance.mat');

%% Extract variables from the structure
subject_id = categorical(data_struct.subject_id);  % Convert subject_id to categorical
target_vergence = data_struct.target_vergence;
measured_vergence = data_struct.measured_vergence;

% Check lengths of variables
n = numel(subject_id);
m1 = numel(target_vergence);
m2 = numel(measured_vergence);

% Check if all lengths are the same
if n == m1 && n == m2
    % Create the data with row vectors
    data = table(subject_id', target_vergence', measured_vergence', ...
        'VariableNames', {'subject', 'target_vergence', 'measured_vergence'});

    % Display the data
%     disp(data);
else
    error('Number of rows in variables subject_id, target_vergence, and measured_vergence must be the same.');
end

%% Linear Mixed Effects Model
formula = 'measured_vergence ~ target_vergence + (1| subject)';
% formula = 'measured_vergence ~ target_vergence + (target_vergence| subject)';

lme = fitlme(data, formula,...
    'FitMethod', 'REML', 'Verbose', true);

disp(lme)

%% Extract Random Effects
re = randomEffects(lme);

% Display the random effects
disp('Random Effects (Intercepts) by Subject:');
disp(re);


%% Extract variables from the structure
subject_id = categorical(data_struct.subject_id);  % Convert subject_id to categorical
target_vergence = data_struct.target_vergence;

% Define the random intercepts for each subject
intercepts = struct('s001', -0.9395, 's002', -1.0802, 's003', 0.8065, 's007', 0.3938, 's008', 0.8194);

% Initialize an array to store the adjusted vergence values
adjusted_vergence = measured_vergence;

% Loop through each subject and adjust the vergence values
for i = 1:length(subject_id)
subj = char(subject_id(i));
adjusted_vergence(i) = measured_vergence(i) - intercepts.(subj);
end

% Check lengths of variables
n = numel(subject_id);
m1 = numel(target_vergence);
m2 = numel(adjusted_vergence);

% Check if all lengths are the same
if n == m1 && n == m2
    % Create the data with row vectors
    data_adjusted = table(subject_id', target_vergence', adjusted_vergence', ...
        'VariableNames', {'subject', 'target_vergence', 'adjusted_vergence'});

    % Display the data
%     disp(data);
else
    error('Number of rows in variables subject_id, target_vergence, and adjusted_vergence must be the same.');
end

%% Linear Mixed Effects Model
formula = 'adjusted_vergence ~ target_vergence + (1| subject)';
% formula = 'adjusted_vergence ~ target_vergence + (target_vergence| subject)';

lme_adjusted = fitlme(data_adjusted, formula,...
    'FitMethod', 'REML', 'Verbose', true);

disp(lme_adjusted)

%% Extract Random Effects
re = randomEffects(lme_adjusted);

% Display the random effects
disp('Random Effects (Intercepts) by Subject:');
disp(re);