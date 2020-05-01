def train(dataloader, model, criterion, optimizer, auc, device):
    model.train()
    
    prev_auc = []
    
    print('Training: \n')
    
    for i, data in enumerate(dataloader):
        inputs, target = data['inputs'], data['targets']
        
        inputs = inputs.to(device)
        target = torch.tensor(target, dtype=torch.float)
        target = target.to(device)
        #print(inputs.size(), target.size())
        optimizer.zero_grad()
        output = model(inputs)
        #print(f"Output: {output.squeeze(1).size()}")
        #print(f"Target: {target.size()}")
        loss = criterion(output, target)
        auc_score = auc(target.detach().cpu().numpy(), output.detach().cpu().numpy())
        #print(target.detach().cpu().numpy().shape)
        loss.backward()
        optimizer.step()
        
        if i % 400 == 0:
            print(f"bi: {i},  loss: {loss.item():.4f},  auc: {auc_score:.4f}")
        
            if len(prev_auc) == 0:
                prev_auc.append(auc_score)

            if (len(prev_auc) > 0) and (auc_score > max(prev_auc)):
                prev_auc.append(auc_score)
                torch.save(model, f'model{len(prev_auc)}.pth')
        
    return loss.item()


def evaluate(dataloader, model, criterion, optimizer, auc, device):
    model.eval()
    
    scores = []
    print('\n')
    print('Validation: \n')
    for i, data in enumerate(dataloader):
        inputs, target = data['inputs'], data['targets']
        
        inputs = inputs.to(device)
        target = torch.tensor(target, dtype=torch.float)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, target)
        auc_score = auc(target.detach().cpu().numpy(), output.detach().cpu().numpy())
        scores.append(auc_score)
        
        if i % 100 == 0:
            print(f"bi: {i},  loss: {loss.item():.4f},  auc: {auc_score:.4f}")

    return loss.item(), np.mean(scores)