function [x_est, P_est, Q_seq] = adaptiveKalman(z, Q0, R, opts)
    z = z(:);
    N = length(z);
    if nargin<4, opts = struct(); end
    if ~isfield(opts,'gamma'),    opts.gamma = 3; end
    if ~isfield(opts,'a_up'),     opts.a_up = 5; end
    if ~isfield(opts,'a_down'),   opts.a_down = 0.95; end
    if ~isfield(opts,'Q_min'),    opts.Q_min = Q0*1e-3; end
    if ~isfield(opts,'Q_max'),    opts.Q_max = Q0*1e3; end
    if ~isfield(opts,'P0'),       opts.P0 = 1; end
    if ~isfield(opts,'returnToQ0'), opts.returnToQ0 = false; end
    if ~isfield(opts,'beta'),     opts.beta = 0.95; end   
    x_est = zeros(N,1);
    P_est = zeros(N,1);
    Q_seq = zeros(N,1);   
    A = 1; H = 1;
    x_prev = z(1);
    P_prev = opts.P0;
    Q_prev = Q0;    
    x_est(1) = x_prev;
    P_est(1) = P_prev;
    Q_seq(1) = Q_prev;

    for k = 2:N        
        x_pred = A * x_prev;
        P_pred = A * P_prev * A + Q_prev;        
        y = z(k) - H * x_pred;
        S = H * P_pred * H + R;        
        if abs(y) > opts.gamma * sqrt(S)            
            Q_curr = min(Q_prev * opts.a_up, opts.Q_max);
        else            
            Q_curr = max(Q_prev * opts.a_down, opts.Q_min);
            if opts.returnToQ0                
                Q_curr = opts.beta * Q_curr + (1-opts.beta) * Q0;
            end
        end        
        P_pred = A * P_prev * A + Q_curr;     
        K = P_pred * H / (H * P_pred * H + R);        
        x_curr = x_pred + K * y;
        P_curr = (1 - K * H) * P_pred;         
        x_est(k) = x_curr;
        P_est(k) = P_curr;
        Q_seq(k) = Q_curr;         
        x_prev = x_curr;
        P_prev = P_curr;
        Q_prev = Q_curr;
    end
end
