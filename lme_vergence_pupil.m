%% Load your data from the .mat file
data = load('data_vergence_pupil.mat');

%% lme vergence and pupil area
% Extract variables from the structure
subject_id = categorical(data.subject_id);  % Convert subject_id to categorical
average_pupil_area = data.average_pupil_area;
measured_vergence = data.measured_vergence;

% Check lengths of variables
n = numel(subject_id);
m1 = numel(average_pupil_area);
m2 = numel(measured_vergence);

% Check if all lengths are the same
if n == m1 && n == m2
    % Create the dataTable with row vectors
    dataTable = table(subject_id', average_pupil_area', measured_vergence', ...
        'VariableNames', {'subject', 'average_pupil_area', 'measured_vergence'});

    % Display the dataTable
%     disp(dataTable);
else
    error('Number of rows in variables subject_id, average_pupil_area, and measured_vergence must be the same.');
end

%% perform linear mixed effects
formula_complex = 'measured_vergence ~ average_pupil_area + (average_pupil_area| subject)';
formula_simple = 'measured_vergence ~ average_pupil_area + (1| subject)';

lme_complex = fitlme(dataTable, formula_complex,...
    'FitMethod', 'ML', 'Verbose', true);
lme_simple = fitlme(dataTable, formula_simple,...
    'FitMethod', 'ML', 'Verbose', true);

disp(lme_complex)
disp(lme_simple)

%% display intercept and slopes
% Extract fixed effects (overall intercept and slope)
fixed_effects = fixedEffects(lme_complex); % [Intercept; Slope]

% Extract random effects
[random_effects, names] = randomEffects(lme_complex); 

disp(fixed_effects)
disp(names)
disp(random_effects)


%% likelihood ratio test
% Use compare function
results = compare(lme_simple, lme_complex, 'CheckNesting', true);

% Display results
disp(results);

% Extract relevant information
LRstat = results.LRStat(end);  % LR statistic is in the last row
LRdf = results.deltaDF;        % DF is in the last row
pValue = results.pValue(end);  % p-value is in the last row

% Display formatted results
fprintf('Likelihood Ratio Statistic: %.4f\n', LRstat);
fprintf('Degrees of Freedom: %d\n', LRdf);
fprintf('p-value: %.4f\n', pValue);

%% likelihood ratio test
% Use compare function
results = compare(lme_simple, lme_complex, 'CheckNesting', true);

% Display results
disp(results);

% Extract relevant information
LRstat = results.LRStat(end);  % LR statistic is in the last row
LRdf = results.deltaDF;        % DF is in the last row
pValue = results.pValue(end);  % p-value is in the last row

% Display formatted results
fprintf('Likelihood Ratio Statistic: %.4f\n', LRstat);
fprintf('Degrees of Freedom: %d\n', LRdf);
fprintf('p-value: %.4f\n', pValue);

%% Spearman Rank Correlation
[Rho, Pval] = corr(dataTable.measured_vergence, dataTable.average_pupil_area, 'Type', 'Spearman');
display([Rho, Pval])

%% lme vergence and pupil diameter
% Extract variables from the structure
subject_id = categorical(data.subject_id);  % Convert subject_id to categorical
average_pupil_diameter_mm = data.average_pupil_diameter_mm;
measured_vergence = data.measured_vergence;

% Check lengths of variables
n = numel(subject_id);
m1 = numel(average_pupil_diameter_mm);
m2 = numel(measured_vergence);

% Check if all lengths are the same
if n == m1 && n == m2
    % Create the dataTable with row vectors
    dataTable = table(subject_id', average_pupil_diameter_mm', measured_vergence', ...
        'VariableNames', {'subject', 'average_pupil_diameter_mm', 'measured_vergence'});

    % Display the dataTable
%     disp(dataTable);
else
    error('Number of rows in variables subject_id, average_pupil_diameter_mm, and measured_vergence must be the same.');
end

%% perform linear mixed effects
formula_complex = 'measured_vergence ~ average_pupil_diameter_mm + (average_pupil_diameter_mm| subject)';
formula_simple = 'measured_vergence ~ average_pupil_diameter_mm + (1| subject)';

lme_complex_diam = fitlme(dataTable, formula_complex,...
    'FitMethod', 'ML', 'Verbose', true);
lme_simple_diam = fitlme(dataTable, formula_simple,...
    'FitMethod', 'ML', 'Verbose', true);

disp(lme_complex_diam)
disp(lme_simple_diam)

%% display intercept and slopes
% Extract fixed effects (overall intercept and slope)
fixed_effects_diam = fixedEffects(lme_complex_diam); % [Intercept; Slope]

% Extract random effects
[random_effects_diam, names_diam] = randomEffects(lme_complex_diam); 

disp(fixed_effects_diam)
disp(names_diam)
disp(random_effects_diam)

%% lme vergence and pupil diameter
% Extract variables from the structure
subject_id = categorical(data.subject_id);  % Convert subject_id to categorical
average_pupil_area = data.average_pupil_area;
left_pog_x_deg = data.left_pog_x_deg;
left_pog_y_deg = data.left_pog_y_deg;
right_pog_x_deg = data.right_pog_x_deg;
right_pog_y_deg = data.right_pog_y_deg;

% Check lengths of variables
n = numel(subject_id);
m1 = numel(left_pog_x_deg);
m2 = numel(left_pog_y_deg);
m3 = numel(right_pog_x_deg);
m4 = numel(right_pog_y_deg);
m2 = numel(measured_vergence);

% Check if all lengths are the same
if n == m1 && n == m2
    % Create the dataTable with row vectors
    dataTable = table(subject_id', average_pupil_area', left_pog_x_deg', left_pog_y_deg', right_pog_x_deg', right_pog_y_deg', ...
        'VariableNames', {'subject', 'average_pupil_area', 'left_pog_x_deg', 'left_pog_y_deg', 'right_pog_x_deg', 'right_pog_y_deg'});

    % Display the dataTable
%     disp(dataTable);
else
    error('Number of rows in variables subject_id, average_pupil_diameter_mm, and measured_vergence must be the same.');
end

%% perform linear mixed effects
formula_left_x = 'left_pog_x_deg ~ average_pupil_area + (average_pupil_area| subject)';
formula_left_y = 'left_pog_y_deg ~ average_pupil_area + (average_pupil_area| subject)';
formula_right_x = 'right_pog_x_deg ~ average_pupil_area + (average_pupil_area| subject)';
formula_right_y = 'right_pog_y_deg ~ average_pupil_area + (average_pupil_area| subject)';

lme_left_x = fitlme(dataTable, formula_left_x, 'FitMethod', 'ML', 'Verbose', true);
lme_left_y = fitlme(dataTable, formula_left_y, 'FitMethod', 'ML', 'Verbose', true);
lme_right_x = fitlme(dataTable, formula_right_x, 'FitMethod', 'ML', 'Verbose', true);
lme_right_y = fitlme(dataTable, formula_right_y, 'FitMethod', 'ML', 'Verbose', true);

%% Store all models in a cell array for easy iteration
models = {lme_left_x, lme_left_y, lme_right_x, lme_right_y};
model_names = {'Left X', 'Left Y', 'Right X', 'Right Y'};

% Initialize sto%% Store all models in a cell array for easy iteration
models = {lme_left_x, lme_left_y, lme_right_x, lme_right_y};
model_names = {'Left X', 'Left Y', 'Right X', 'Right Y'};

% Initialize storage for all models
all_fixed_intercepts = table();
all_fixed_slopes = table();
all_random_intercepts = table();
all_random_slopes = table();

for i = 1:length(models)
    lme = models{i};
    model_name = model_names{i};
    
    % Extract fixed effects (population-level)
    fixed_effects = fixedEffects(lme);
    fixed_intercept = fixed_effects(1);
    fixed_slope = fixed_effects(2);

    % Store fixed effects in a table
    all_fixed_intercepts.(model_name) = fixed_intercept;
    all_fixed_slopes.(model_name) = fixed_slope;

    % Extract random effects (subject-specific)
    random_effects = randomEffects(lme); % random_effects is a numeric vector

    % Separate the intercepts and slopes assuming alternating values
    random_intercepts = random_effects(1:2:end); % Odd indices (1, 3, 5, ...)
    random_slopes = random_effects(2:2:end);    % Even indices (2, 4, 6, ...)

    % Store random intercepts and slopes in tables
    random_intercepts_table = table(random_intercepts, 'VariableNames', {model_name});
    random_slopes_table = table(random_slopes, 'VariableNames', {model_name});

    % Append the new values to the respective result tables
    all_random_intercepts = [all_random_intercepts, random_intercepts_table];
    all_random_slopes = [all_random_slopes, random_slopes_table];
end

% Display results
disp('Fixed Intercepts:');
disp(all_fixed_intercepts);

disp('Fixed Slopes:');
disp(all_fixed_slopes);

disp('Random Intercepts:');
disp(all_random_intercepts);

disp('Random Slopes:');
disp(all_random_slopes);

