function loss = tv_loss(w,type)


loss1 = abs(w(1:end-1,:) - w(2:end,:));
loss2 = abs(w(:,1:end-1) - w(:,2:end));



switch type
    case 'sum'
        loss = sum(loss1(:)) + sum(loss2(:));
        loss = sum(loss(:));
    case 'mean'
        loss = mean(loss1(:)) + mean(loss2(:));
        loss = mean(loss(:));
    otherwise 
        error('type must be sum or mean')
end
end