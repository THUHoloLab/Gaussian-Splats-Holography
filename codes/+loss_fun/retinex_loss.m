function loss = retinex_loss(x1,x2,type)
w = x1 - x2;

loss1 = [w(2:end,:) - w(1:end-1,:); w(1,:) - w(end,:)];
loss2 = [w(:,2:end) - w(:,1:end-1), w(:,1) - w(:,end)];

loss = sqrt(abs(loss1).^2 + abs(loss2).^2 + 1e-5);

switch type
    case 'sum'
        loss = sum(loss(:));
    case 'mean'
        loss = mean(loss(:));
    otherwise 
        error('type must be sum or mean')
end
end