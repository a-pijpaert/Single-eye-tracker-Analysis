% Load your data from the .mat file
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