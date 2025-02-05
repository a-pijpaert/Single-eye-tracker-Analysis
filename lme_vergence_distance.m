%%% perform linear mixed effects on vergence data

data_struct = load('data/vergence_distance.mat');

% Extract variables from the structure
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

lme = fitlme(data, formula,...
    'FitMethod', 'REML', 'Verbose', true);

disp(lme)

%% Extract Random Effects
re = randomEffects(lme);

% Display the random effects
disp('Random Effects (Intercepts) by Subject:');
disp(re);
