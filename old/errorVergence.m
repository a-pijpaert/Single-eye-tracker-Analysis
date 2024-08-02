% error when assuming a fixed IOD and center of rotation

iods = 50:1:70;
pTarget = [0, 0, 0];

vergence_angles = zeros(length(iods), 1);
for i=1:length(iods)
    iod = iods(i);
    pLeft = [iod/2, 0, 650];
    pRight = [-iod/2, 0, 650];
    vergence_angles(i) = calculateVergence(iod, pTarget, pLeft, pRight);
end

% Perform linear regression
p = polyfit(iods, vergence_angles', 1);  % p(1) is the slope, p(2) is the intercept

% Display the linear relation
fprintf('Linear relation: Vergence = %.4f * IOD + %.4f\n', p(1), p(2));

% Plot the data and the linear fit
figure;
plot(iods, vergence_angles, 'o', 'DisplayName', 'Data');
hold on;
plot(iods, polyval(p, iods), '-', 'DisplayName', 'Linear Fit');
xlabel('IOD (mm)');
ylabel('Vergence Angle (degrees)');
title('Linear Relation between IOD and Vergence');
legend show;
grid on;



% iod = 60;
% pTarget = [0, 0, 0];
% z_distances_eye = 640:1:660;
% 
% vergence_angles = zeros(length(z_distances_eye), 1);
% for i=1:length(z_distances_eye)
%     z_distance_eye = z_distances_eye(i);
%     pLeft = [iod/2, 0, z_distance_eye];
%     pRight = [-iod/2, 0, z_distance_eye];
%     vergence_angles(i) = calculateVergence(iod, pTarget, pLeft, pRight);
% end
% 
% % Perform linear regression
% p = polyfit(iods, vergence_angles', 1);  % p(1) is the slope, p(2) is the intercept
% 
% % Display the linear relation
% fprintf('Linear relation: Vergence = %.4f * z_distance_eye + %.4f\n', p(1), p(2));
% 
% % Plot the data and the linear fit
% figure;
% plot(z_distances_eye, vergence_angles, 'o', 'DisplayName', 'Data');
% hold on;
% plot(z_distances_eye, polyval(p, iods), '-', 'DisplayName', 'Linear Fit');
% xlabel('z_distance_eye (mm)');
% ylabel('Vergence Angle (degrees)');
% title('Linear Relation between z_distance_eye and Vergence');
% legend show;
% grid on;