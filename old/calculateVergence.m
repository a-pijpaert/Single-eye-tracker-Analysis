function vergence = calculateVergence(iod, pTarget, pLeft, pRight)
    % calculateVergence calculates the vergence angle given the IOD, target point,
    % left eye point, and right eye point.
    %
    % Inputs:
    %   - iod: Interocular distance (in mm or any consistent unit)
    %   - pTarget: 1x3 vector representing the target point coordinates [x y z]
    %   - pLeft: 1x3 vector representing the left eye point coordinates [x y z]
    %   - pRight: 1x3 vector representing the right eye point coordinates [x y z]
    %
    % Output:
    %   - vergence: Vergence angle in degrees

    % Calculate distances from the target point to each eye
    dl = norm(pLeft - pTarget);
    dr = norm(pRight - pTarget);

    % Calculate vergence angle using the cosine rule
    vergence = acosd((dl^2 + dr^2 - iod^2) / (2 * dl * dr));
end
