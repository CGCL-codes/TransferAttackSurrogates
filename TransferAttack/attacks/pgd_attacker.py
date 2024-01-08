import torch


class PGDAttacker():
    def __init__(self, radius, steps, step_size, random_start, norm_type, ascending=True):
        self.norm_type = norm_type
        if self.norm_type == 'l-infty':
            self.radius = radius / 255.
            self.steps = steps
            self.step_size = step_size / 255.
            self.random_start = random_start
            self.ascending = ascending
        elif self.norm_type == 'l2':
            self.radius = radius
            self.steps = steps
            self.step_size = step_size
            self.random_start = random_start
            self.ascending = ascending

    def perturb(self, model, criterion, x, y):
        if self.steps==0 or self.radius==0:
            return x.clone()

        adv_x = x.clone()
        if self.random_start:
            if self.norm_type == 'l-infty':
                adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius
            else:
                adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius / self.steps
            self._clip_(adv_x, x)

        ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        model.eval()
        for pp in model.parameters():
            pp.requires_grad = False

        for step in range(self.steps):
            adv_x.requires_grad_()
            _y = model(adv_x)
            loss = criterion(_y, y)
            grad = torch.autograd.grad(loss, adv_x)[0]

            with torch.no_grad():
                if not self.ascending: grad.mul_(-1)

                if self.norm_type == 'l-infty':
                    adv_x.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    if self.norm_type == 'l2':
                        grad_norm = (grad.reshape(grad.shape[0],-1)**2).sum(dim=1).sqrt()
                    elif self.norm_type == 'l1':
                        grad_norm = grad.reshape(grad.shape[0],-1).abs().sum(dim=1)
                    grad_norm = grad_norm.reshape( -1, *( [1] * (len(x.shape)-1) ) )
                    scaled_grad = grad / (grad_norm + 1e-10)
                    adv_x.add_(scaled_grad, alpha=self.step_size)

                self._clip_(adv_x, x)
                
        ''' reopen autograd of model after pgd '''
        for pp in model.parameters():
            pp.requires_grad = True

        return adv_x.data

    def _clip_(self, adv_x, x):
        adv_x -= x
        if self.norm_type == 'l-infty':
            adv_x.clamp_(-self.radius, self.radius)
        else:
            if self.norm_type == 'l2':
                norm = (adv_x.reshape(adv_x.shape[0],-1)**2).sum(dim=1).sqrt()
            elif self.norm_type == 'l1':
                norm = adv_x.reshape(adv_x.shape[0],-1).abs().sum(dim=1)
            norm = norm.reshape( -1, *( [1] * (len(x.shape)-1) ) )
            adv_x /= (norm + 1e-10)
            adv_x *= norm.clamp(max=self.radius)
        adv_x += x
        adv_x.clamp_(0, 1)



'''
for k in tqdm_notebook(range(0,num_batches)):
    batch_size_cur = min(batch_size,len(image_id_list) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur,3,img_size,img_size).to(device)
    delta = torch.zeros_like(X_ori,requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
    labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
    grad_pre = 0
    prev = float('inf')
    for t in range(max_iterations):
        logits = model_2(norm(DI(X_ori + delta))) #DI
        loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
        loss.backward()
        grad_c = delta.grad.clone()
        grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3) #TI
        grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre #MI
        grad_pre = grad_a
        delta.grad.zero_()
        delta.data = delta.data - lr * torch.sign(grad_a)
        delta.data = delta.data.clamp(-epsilon / 255,epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0,1)) - X_ori
        if t % 20 == 19:
            pos[0,t // 20] = pos[0,t // 20] + sum(torch.argmax(model_1(norm(X_ori + delta)),dim=1) == labels).cpu().numpy()
            pos[1,t // 20] = pos[1,t // 20] + sum(torch.argmax(model_3(norm(X_ori + delta)),dim=1) == labels).cpu().numpy()
            pos[2,t // 20] = pos[2,t // 20] + sum(torch.argmax(model_4(norm(X_ori + delta)),dim=1) == labels).cpu().numpy()
torch.cuda.empty_cache()

'''
