function R = euler2rotm(euler_angles, order)
%EULER2ROTM Converts Euler angles to rotation matrix
%   R = EULER2ROTM(EULER_ANGLES, ORDER) converts the Euler angles in the
%   vector EULER_ANGLES to the corresponding rotation matrix R, using the
%   specified rotation order ORDER.
%
%   EULER_ANGLES is a 1x3 vector containing the Euler angles in radians.
%   ORDER is a string specifying the order of rotations, e.g., 'ZYX'.
%
%   The rotation matrix R is a 3x3 orthogonal matrix that rotates a vector
%   by the specified Euler angles in the given order.

% Extract Euler angles
alpha = euler_angles(1);
beta = euler_angles(2);
gamma = euler_angles(3);

% Define rotation matrices for each axis
Rx = [1 0 0; 0 cos(alpha) -sin(alpha); 0 sin(alpha) cos(alpha)];
Ry = [cos(beta) 0 sin(beta); 0 1 0; -sin(beta) 0 cos(beta)];
Rz = [cos(gamma) -sin(gamma) 0; sin(gamma) cos(gamma) 0; 0 0 1];

% Combine rotations based on the specified order
switch order
    case 'XYZ'
        R = Rx * Ry * Rz;
    case 'XZY'
        R = Rx * Rz * Ry;
    case 'YXZ'
        R = Ry * Rx * Rz;
    case 'YZX'
        R = Ry * Rz * Rx;
    case 'ZXY'
        R = Rz * Rx * Ry;
    case 'ZYX'
        R = Rz * Ry * Rx;
    otherwise
        error('Invalid rotation order specified.');
end
end