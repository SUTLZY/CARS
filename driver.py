import copy

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

from attention_net import AttentionNet
from runner import RLRunner
from parameters import *
#from VAE.VAE_net import Context_Model, Encoder, Decoder
from torch.optim import Adam
from torch.nn import functional as F

from torch.autograd import Variable
from wae_model import *


#ray.init()
ray.init(ignore_reinit_error=True)
print("Welcome to PRM-AN!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

global_step = None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def writeToTensorBoard(writer, tensorboardData, curr_episode, plotMeans=True):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    if plotMeans == True:
        tensorboardData = np.array(tensorboardData)
        tensorboardData = list(np.nanmean(tensorboardData, axis=0))
        metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace']
        # reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr = tensorboardData
        if USE_VAE:
            if NEXT_OB:
                reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, reproduction_loss, remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr = tensorboardData
            else:
                reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, reproduction_loss, KLD, remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr = tensorboardData
        elif USE_VAE_A:
            reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, reproduction_loss, KLD, prediction_loss, remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr = tensorboardData
        elif USE_WAE:
            if Aug_S:
                reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, reproduction_loss, mmd_loss, remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr = tensorboardData
            else:
                reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, reproduction_loss, mmd_loss, prediction_loss, remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr = tensorboardData
        else:
            reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr = tensorboardData
    else:
        reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr = tensorboardData

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policyLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Value Loss', scalar_value=valueLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Grad Norm', scalar_value=gradNorm, global_step=curr_episode)
    if USE_VAE:
        writer.add_scalar(tag='Losses/Reproduction Loss', scalar_value=reproduction_loss, global_step=curr_episode)
        if not NEXT_OB:
            writer.add_scalar(tag='Losses/KLD', scalar_value=KLD, global_step=curr_episode)
    if USE_WAE:
        if not Aug_S:
            writer.add_scalar(tag='Losses/prediction_loss',   scalar_value=prediction_loss,   global_step=curr_episode)
        writer.add_scalar(tag='Losses/Reproduction Loss', scalar_value=reproduction_loss, global_step=curr_episode)
        writer.add_scalar(tag='Losses/mmd_loss',          scalar_value=mmd_loss,          global_step=curr_episode)
        #writer.add_scalar(tag='Losses/prediction_loss',   scalar_value=prediction_loss,   global_step=curr_episode)
        #writer.add_scalar(tag='Perf/LR',                  scalar_value=lr,                global_step=curr_episode)
    if USE_VAE_A:
        writer.add_scalar(tag='Losses/Reproduction Loss', scalar_value=reproduction_loss, global_step=curr_episode)
        writer.add_scalar(tag='Losses/KLD',               scalar_value=KLD,               global_step=curr_episode)
        writer.add_scalar(tag='Losses/prediction_loss',   scalar_value=prediction_loss,   global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Returns', scalar_value=returns, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Remain Budget', scalar_value=remain_budget, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/RMSE', scalar_value=RMSE, global_step=curr_episode)
    writer.add_scalar(tag='Perf/F1 Score', scalar_value=F1, global_step=curr_episode)
    writer.add_scalar(tag='GP/MI', scalar_value=MI, global_step=curr_episode)
    writer.add_scalar(tag='GP/Delta Cov Trace', scalar_value=dct, global_step=curr_episode)
    writer.add_scalar(tag='GP/Cov Trace', scalar_value=cov_tr, global_step=curr_episode)

def make_history_buffer(len_history, node_input):
    batch_size = len(node_input)
    sample_size = len(node_input[0])
    history_node_inputs = torch.zeros(batch_size, sample_size, 4 * len_history)
    for i in range(batch_size-1):
        if i >= len_history:
            history_node_inputs[i+1] = torch.cat(node_input[(i+1) - len_history:(i+1)], dim=-1)
        else:
            history_node_inputs[i+1][:, -(i+1)*4:] = torch.cat(node_input[:i + 1], dim=-1)
    
    return history_node_inputs

def loss_function(x, x_hat, mean, log_var):
    #reproduction_loss = nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction='sum')
    #reproduction_loss = F.mse_loss(x_hat, x)
    #print("00001===================================x.size() = ", x.size())
    #print("00002===================================log_var.size() = ", log_var.size())
    #reconstruction_loss = torch.mean(torch.square(x - x_hat).sum(dim=1))
    reconstruction_loss = torch.mean(torch.pow(x - x_hat, 2), dim=-1)
    #reconstruction_loss = torch.mean(torch.pow(x - x_hat, 2), dim=(1, 2))
    #print("00002===================================reconstruction_loss.size() = ", reconstruction_loss.size())
    kld                 = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp(), dim=-1)
    #print("00003===================================kld.size() = ", kld.size())
    #kld                 = kld.mean()
    #print("00004===================================kld.size() = ", kld.size())
    
    return reconstruction_loss, kld


def main():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_DEVICE)[1:-1]
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    set_seed(SEED)
    x_dim = LEN_HISTORY * 4
    global_network = AttentionNet(INPUT_DIM, EMBEDDING_DIM, LEN_HISTORY*4).to(device)
    
    # global_network.share_memory()
    global_optimizer = optim.Adam(global_network.parameters(), lr=LR)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=DECAY_STEP, gamma=0.96)
    # Automatically logs gradients of pytorch model
    #wandb.watch(global_network, log_freq = SUMMARY_WINDOW)
    #context_optimizer = Adam(contex_network.parameters(), lr=1e-3)
    

    best_perf = 900
    curr_episode = 0
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth')
        global_network.load_state_dict(checkpoint['model'])
        global_optimizer.load_state_dict(checkpoint['optimizer'])
        lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']
        print("curr_episode set to ", curr_episode)

        best_model_checkpoint = torch.load(model_path + '/best_model_checkpoint.pth')
        best_perf = best_model_checkpoint['best_perf']
        print('best performance so far:', best_perf)
        print(global_optimizer.state_dict()['param_groups'][0]['lr'])

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get initial weigths
    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        global_network.to(device)
    else:
        weights = global_network.state_dict()

    # launch the first job on each runner
    dp_model = nn.DataParallel(global_network)
    #dp_model_context = nn.DataParallel(contex_network)

    jobList = []
    sample_size = np.random.randint(200,400)
    #print("000004===================================sample_size = ", sample_size)
    for i, meta_agent in enumerate(meta_agents):
        jobList.append(meta_agent.job.remote(weights, curr_episode, BUDGET_RANGE, sample_size, SAMPLE_LENGTH))
        curr_episode += 1
    metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace']
    tensorboardData = []
    trainingData = []
    experience_buffer = []
    #history_buffer = []
    for i in range(LEN_EXP_BUFFER):
        experience_buffer.append([])

    try:
        while True:
            # wait for any job to be completed
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
            # get the results
            #jobResults, metrics, info = ray.get(done_id)[0]
            done_jobs = ray.get(done_id)
            random.shuffle(done_jobs)
            #done_jobs = list(reversed(done_jobs))
            perf_metrics = {}
            for n in metric_name:
                perf_metrics[n] = []
            for job in done_jobs:
                jobResults, metrics, info = job
                #print("000001===================================jobResults[0][0].size() = ", jobResults[0][0].size())
                #print("000001===================================len(jobResults[0]) = ", len(jobResults[0]))
                #print("000002===================================len(jobResults[0][0]) = ", len(jobResults[0][0]))
                #print("000003===================================len(jobResults[0][0][0]) = ", len(jobResults[0][0][0]))
                #print("000001===================================jobResults.type = ", jobResults.type)
                #history_buffer = jobResults[0]
                #experience_buffer[13] += make_history_buffer(LEN_HISTORY, jobResults[0])
                #print("000002===================================LEN_EXP_BUFFER = ", LEN_EXP_BUFFER)
                for i in range(14):
                    if not NEXT_OB:
                        experience_buffer[i] += jobResults[i]
                    else:
                        if i == 0:
                            experience_buffer[14] += jobResults[i][1:]  # 获取next_ob
                            experience_buffer[0] += jobResults[i][:-1] # 获取当前ob
                        else:
                            jobResults[i].pop()
                            experience_buffer[i] += jobResults[i]
                #for i in range(len(experience_buffer)):
                #    print(f"000003===================================len(experience_buffer[{i}]) = ", len(experience_buffer[i]))
                    
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])
            
            if np.mean(perf_metrics['cov_trace']) < best_perf and curr_episode % 32 == 0:
                best_perf = np.mean(perf_metrics['cov_trace'])
                print('Saving best model', end='\n')
                checkpoint = {"model": global_network.state_dict(),
                              "optimizer": global_optimizer.state_dict(),
                              "episode": curr_episode,
                              "lr_decay": lr_decay.state_dict(),
                              "best_perf": best_perf}
                path_checkpoint = "./" + model_path + "/best_model_checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')

            update_done = False
            while len(experience_buffer[0]) >= BATCH_SIZE:
                rollouts = copy.deepcopy(experience_buffer)
                for i in range(len(rollouts)):
                    rollouts[i] = rollouts[i][:BATCH_SIZE]
                for i in range(len(experience_buffer)):
                    experience_buffer[i] = experience_buffer[i][BATCH_SIZE:]
                if len(experience_buffer[0]) < BATCH_SIZE:
                    update_done = True
                if update_done:
                    experience_buffer = []
                    for i in range(LEN_EXP_BUFFER):
                        experience_buffer.append([])
                    sample_size = np.random.randint(200,400)

                node_inputs_batch = torch.stack(rollouts[0], dim=0) # (batch,sample_size+2,2)
                #print("000001===================================node_inputs_batch.size() = ", node_inputs_batch.size())
                #print("000002===================================rollouts[0][0].size() = ", rollouts[0][0].size())
                #print("000003===================================len(rollouts[0]) = ", len(rollouts[0]))
                edge_inputs_batch = torch.stack(rollouts[1], dim=0) # (batch,sample_size+2,k_size)
                current_inputs_batch = torch.stack(rollouts[2], dim=0) # (batch,1,1)
                action_batch = torch.stack(rollouts[3], dim=0) # (batch,1,1)
                value_batch = torch.stack(rollouts[4], dim=0) # (batch,1,1)
                reward_batch = torch.stack(rollouts[5], dim=0) # (batch,1,1)
                value_prime_batch = torch.stack(rollouts[6], dim=0) # (batch,1,1)
                target_v_batch = torch.stack(rollouts[7])
                budget_inputs_batch = torch.stack(rollouts[8], dim=0)
                LSTM_h_batch = torch.stack(rollouts[9])
                LSTM_c_batch = torch.stack(rollouts[10])
                mask_batch = torch.stack(rollouts[11])
                pos_encoding_batch = torch.stack(rollouts[12])
                context_batch = torch.stack(rollouts[13])
                if NEXT_OB:
                    next_ob_batch = torch.stack(rollouts[14])

                if device != local_device:
                    node_inputs_batch = node_inputs_batch.to(device)
                    edge_inputs_batch = edge_inputs_batch.to(device)
                    current_inputs_batch = current_inputs_batch.to(device)
                    action_batch = action_batch.to(device)
                    value_batch = value_batch.to(device)
                    reward_batch = reward_batch.to(device)
                    value_prime_batch = value_prime_batch.to(device)
                    target_v_batch = target_v_batch.to(device)
                    budget_inputs_batch = budget_inputs_batch.to(device)
                    LSTM_h_batch = LSTM_h_batch.to(device)
                    LSTM_c_batch = LSTM_c_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    pos_encoding_batch = pos_encoding_batch.to(device)
                    context_batch = context_batch.to(device)
                    if NEXT_OB:
                        next_ob_batch = next_ob_batch.to(device)

                # PPO
                with torch.no_grad():
                    if USE_VAE:
                        if NEXT_OB:
                            logp_list, value, _, _, _ = global_network(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, context_batch, mask_batch)
                        else:
                            logp_list, value, _, _, _, _, _ = global_network(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, context_batch, mask_batch)
                    elif USE_WAE:
                        if Aug_S:
                            logp_list, value, _, _, _, _ = global_network(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, context_batch, mask_batch)
                        else:
                            logp_list, value, _, _, _, _, _ = global_network(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, context_batch, mask_batch)
                    elif USE_VAE_A:
                        logp_list, value, _, _, _, _, _, _ = global_network(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, context_batch, mask_batch)
                    else:
                        logp_list, value, _, _ = global_network(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, mask_batch)
                old_logp = torch.gather(logp_list, 1 , action_batch.squeeze(1)).unsqueeze(1) # (batch_size,1,1)
                advantage = (reward_batch + GAMMA*value_prime_batch - value_batch) # (batch_size, 1, 1)
                #advantage = target_v_batch - value_batch

                entropy = (logp_list*logp_list.exp()).sum(dim=-1).mean()

                scaler = GradScaler()

                for i in range(8):
                    with autocast():
                        if USE_VAE:
                            if NEXT_OB:
                                logp_list, value, _, _, x_hat = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, context_batch, mask_batch)
                            else:
                                logp_list, value, _, _, x_hat, mean, log_var = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, context_batch, mask_batch)
                        elif USE_WAE:
                            if Aug_S:
                                logp_list, value, _, _, x_recon, z_real = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, context_batch, mask_batch)
                            else:
                                logp_list, value, _, _, x_recon, z_real, next_ob = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, context_batch, mask_batch)
                            
                        elif USE_VAE_A:
                            logp_list, value, _, _, x_recon, mean, log_var, next_ob = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, context_batch, mask_batch)
                        else:
                            logp_list, value, _, _ = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, mask_batch)
                        logp = torch.gather(logp_list, 1, action_batch.squeeze(1)).unsqueeze(1)
                        ratios = torch.exp(logp-old_logp.detach())
                        surr1 = ratios * advantage.detach()
                        surr2 = torch.clamp(ratios, 1-0.2, 1+0.2) * advantage.detach()
                        policy_loss = -torch.min(surr1, surr2)
                        policy_loss = policy_loss.mean()
                        
                        '''
                        stats = DataStatistics(20)
                        data  = x_hat
                        stats.update(data.detach().cpu().numpy())
                        # 获取上下界
                        min_values, max_values = stats.get_bounds()
                        print("0000002===================Min values:", min_values)
                        print("0000003===================Max values:", max_values)
                        '''
                        
                        if USE_VAE:
                            if NEXT_OB:
                                #print("000002===================================next_ob_batch.size = ", next_ob_batch.size())
                                #print("000003===================================x_hat.size = ", x_hat.size())
                                reproduction_loss = torch.mean(torch.pow(next_ob_batch - x_hat, 2), dim=(1, 2))
                                reproduction_loss = torch.mean(reproduction_loss)
                            else:
                                reproduction_loss, kld = loss_function(context_batch, x_hat, mean, log_var)
                                reproduction_loss = reproduction_loss.mean()
                                kld = kld.mean()
                            #print("000001===================================reproduction_loss.size = ", reproduction_loss.size())
                        
                        if USE_VAE_A:
                            reproduction_loss, kld = loss_function(context_batch, x_recon, mean, log_var)
                            kld = kld.mean()
                            reproduction_loss = reproduction_loss.mean()
                            prediction_loss = torch.mean(torch.pow(next_ob_batch - next_ob, 2), dim=-1)
                            #prediction_loss = torch.mean(torch.pow(next_ob_batch - next_ob, 2), dim=(1, 2))
                            prediction_loss = torch.mean(prediction_loss)
                            
                            vae_total_loss = reproduction_loss + 50 * kld + prediction_loss
                        
                        if USE_WAE:
                            if Aug_S:
                                recon_loss = torch.mean(torch.pow(x_recon - context_batch, 2), dim=-1)
                                recon_loss = recon_loss.mean()
                                
                                z_true = Variable(torch.randn_like(z_real))
                                z_true = z_true.to(device)

                                mmd_loss = mmd(z_true, z_real).mean()

                                wae_total_loss = recon_loss + MMD_LOSS_COEF*mmd_loss
                            else:
                                #criterion = nn.MSELoss()
                                #recon_loss = criterion(x_recon, context_batch).mean()
                                recon_loss = torch.mean(torch.pow(x_recon - context_batch, 2), dim=-1)
                                recon_loss = recon_loss.mean()
                                '''
                                #z_real = z_real.view(-1, 4)  # (batch_size,sample_size,feature_size)==>(batch_size*sample_size,feature_size)
                                #z_real_size0 = z_real.size()[0] # =batch_size*sample_size
                                #sigma = 1
                                #z_fake = Variable(torch.randn(z_real_size0, 20) * sigma)
                                #if torch.cuda.is_available():
                                #    z_fake = z_fake.cuda()
                                #mmd_loss = imq_kernel(z_real, z_fake, h_dim=20)
                                #mmd_loss = mmd_loss / z_real_size0
                                '''
                                z_true = Variable(torch.randn_like(z_real))
                                z_true = z_true.to(device)
                                #z_true = torch.randn_like(z_real).to(device)
                                mmd_loss = mmd(z_true, z_real).mean()
                                prediction_loss = torch.mean(torch.pow(next_ob_batch - next_ob, 2), dim=-1)
                                #prediction_loss = torch.mean(torch.pow(next_ob_batch - next_ob, 2), dim=(1, 2))
                                prediction_loss = torch.mean(prediction_loss)
                                wae_total_loss = recon_loss + MMD_LOSS_COEF*mmd_loss + 0.1*prediction_loss
                            
                        mse_loss = nn.MSELoss()
                        value_loss = mse_loss(value, target_v_batch).mean()
                        entropy_loss = (logp_list * logp_list.exp()).sum(dim=-1).mean()

                        if USE_VAE:
                            if NEXT_OB:
                                loss = policy_loss + 0.5*value_loss + 0.0*entropy_loss + reproduction_loss
                            else:
                                loss = policy_loss + 0.5*value_loss + 0.0*entropy_loss + reproduction_loss #+ kld
                        if USE_WAE:
                            loss = policy_loss + 0.5*value_loss + 0.0*entropy_loss + wae_total_loss
                        if USE_VAE_A:
                            loss = policy_loss + 0.5*value_loss + 0.0*entropy_loss + vae_total_loss
                        else:
                            loss = policy_loss + 0.5*value_loss + 0.0*entropy_loss
                        
                        
                    global_optimizer.zero_grad()
                    # loss.backward()
                    scaler.scale(loss).backward()
                    scaler.unscale_(global_optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=10, norm_type=2)
                    # global_optimizer.step()
                    scaler.step(global_optimizer)
                    scaler.update()
                lr_decay.step()

                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                # data = [reward_batch.mean().item(), value_batch.mean().item(), policy_loss.item(), value_loss.item(),
                #         entropy.item(), grad_norm.item(), target_v_batch.mean().item(), *perf_data]
                if USE_VAE:
                    if NEXT_OB:
                        data = [reward_batch.mean().item(), value_batch.mean().item(), policy_loss.item(), value_loss.item(),
                                entropy.item(), grad_norm.item(), target_v_batch.mean().item(), 
                                reproduction_loss.item(), *perf_data]
                    else:
                        data = [reward_batch.mean().item(), value_batch.mean().item(), policy_loss.item(), value_loss.item(),
                                entropy.item(), grad_norm.item(), target_v_batch.mean().item(), 
                                reproduction_loss.item(), kld.item(), *perf_data]
                
                elif USE_VAE_A:
                    data = [reward_batch.mean().item(), value_batch.mean().item(), policy_loss.item(), value_loss.item(),
                            entropy.item(), grad_norm.item(), target_v_batch.mean().item(), 
                            reproduction_loss.item(), kld.item(), prediction_loss.item(), *perf_data]
                elif USE_WAE:
                    if Aug_S:
                        data = [reward_batch.mean().item(), value_batch.mean().item(), 
                            policy_loss.item(), value_loss.item(),
                            entropy.item(), grad_norm.item(), target_v_batch.mean().item(), 
                            #recon_loss.item(), mmd_loss.item(), prediction_loss.item(), global_optimizer.param_groups[0]['lr'], *perf_data]
                            recon_loss.item(), mmd_loss.item(), *perf_data]
                    else:
                        data = [reward_batch.mean().item(), value_batch.mean().item(), 
                            policy_loss.item(), value_loss.item(),
                            entropy.item(), grad_norm.item(), target_v_batch.mean().item(), 
                            #recon_loss.item(), mmd_loss.item(), prediction_loss.item(), global_optimizer.param_groups[0]['lr'], *perf_data]
                            recon_loss.item(), mmd_loss.item(), prediction_loss.item(), *perf_data]
                #print(f"Epoch {epoch+1}, Learning Rate: {global_optimizer.param_groups[0]['lr']}")
                else:
                    data = [reward_batch.mean().item(), value_batch.mean().item(), policy_loss.item(), value_loss.item(),
                            entropy.item(), grad_norm.item(), target_v_batch.mean().item(), *perf_data]
                trainingData.append(data)

                #experience_buffer = []
                #for i in range(8):
                #    experience_buffer.append([])

            if len(trainingData) >= SUMMARY_WINDOW:
                writeToTensorBoard(writer, trainingData, curr_episode)
                trainingData = []

            # get the updated global weights
            if update_done == True:
                if device != local_device:
                    weights = global_network.to(local_device).state_dict()
                    global_network.to(device)
                else:
                    weights = global_network.state_dict()
            
            jobList = []                                                                                    
            for i, meta_agent in enumerate(meta_agents):                                                    
                jobList.append(meta_agent.job.remote(weights, curr_episode, BUDGET_RANGE, sample_size, SAMPLE_LENGTH))
                curr_episode += 1 
            
            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"model": global_network.state_dict(),
                              "optimizer": global_optimizer.state_dict(),
                              "episode": curr_episode,
                              "lr_decay": lr_decay.state_dict()}
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')
                    
    
    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":
    main()
