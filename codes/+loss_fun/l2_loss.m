function loss = l2_loss(x1,x2,type)

loss = (x1 - x2).^2;

switch type
    case 'sum'
        loss = sum(loss(:));
    case 'mean'
        loss = mean(loss(:));
    otherwise 
        error('type must be sum or mean')
end
end