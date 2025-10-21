from Agent_Architecture import Agent_Model
import pycuda.driver as cuda
import cupy as cp
import numpy as np
import os
import math
import pycuda.autoinit
import time
import matplotlib.pyplot as plt



# Load Kernel
def load_kernel():
    try:
        mod = cuda.module_from_file("/home/alexander/Schreibtisch/Python/KIs/Kernel/matmul_kernel.ptx")       # you can choose a path you like, you can find the PTX kernel in the repository
        print("Modul successfuly loaded")
    except cuda.Error as e:
        print(f"Error trying to load the module: {e}")

    # Kernel-Funktion holen
    matmul_kernel = mod.get_function("matMulOptimized")


    return matmul_kernel

# Get Matrixes
def generate_matrixes(N, M, P):
    
    # Beispiel-Matrizen unterschiedlicher Größe
    
    A = np.random.rand(N, M).astype(np.float32)
    B = np.random.rand(M, P).astype(np.float32)
    C = np.zeros((N, P), dtype=np.float32)


    return A, B, C



def run_kernel(matmul_kernel, N, M, P, A, B, C, tile_size, shared_mem_size, block, grid):
    
    
    A_gpu = cuda.mem_alloc(A.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    C_gpu = cuda.mem_alloc(C.nbytes)

    cuda.memcpy_htod(A_gpu, A)
    cuda.memcpy_htod(B_gpu, B)
    
    # PyCUDA Event zum Messen der Zeit
    start_event = cuda.Event()
    end_event = cuda.Event()
    

    cuda.Context.synchronize()
    
    start_event.record()  # Start der Zeitmessung
    start_event.synchronize()
    
        
    matmul_kernel(
        A_gpu, B_gpu, C_gpu,
        np.int32(N), np.int32(M), np.int32(P),
        np.int32(tile_size),
        block=block,
        grid=grid,
        shared=shared_mem_size
    )
    cuda.Context.synchronize()

    # Stopp der Zeitmessung
    end_event.record()

    # Warten auf das Ende des Events
    end_event.synchronize()

    # Berechnung der Laufzeit in Sekunden
    elapsed_time = start_event.time_till(end_event) 
    
    
    # Ergebnis-Matrix zurück auf die CPU holen
    cuda.memcpy_dtoh(C, C_gpu)
    
    
    return (elapsed_time / 1000), C

# This function is used for both cases; only the parameter names vary depending on how it is called.
def run(tile_size, block_size, N, M, P, A, B, C, FLOP, matmul_kernel, alpha_for_metrik):
    
    shared_mem_size = 2 * tile_size * tile_size * np.float32().itemsize  # Größe des Shared Memorys
    block = (block_size, block_size, 1)
    grid = ((P + tile_size - 1) // tile_size, (N + tile_size - 1) // tile_size)

    # Kernel ausführen
    elapsed_time_in_sec, C = run_kernel(matmul_kernel, N, M, P, A, B, C, tile_size, shared_mem_size, block, grid)


    numbers_r_zero = (C == 0).sum()
        
    FLOPs = FLOP / elapsed_time_in_sec
    TFLOPs = (FLOPs / 1e+12)
    metrik = math.log10(FLOPs / (FLOP ** alpha_for_metrik))


    return metrik, numbers_r_zero, TFLOPs


def exploration_run():

    return







def env_init(N, M, P, max_kernel_size):
    
    
    
        
    state = cp.array([[(N), (M), (P), ((N*M)), (M*P), (N*P)]])

    
    return state


def exploration():
    action = cp.random.rand(1, 2)

    return action



# Trainingparams
episodes = 1000
lr = 1e-4
momentum = 0.3
alpha_for_metrik = 0.85
expl_rate = 0.5
beta_for_ema = 0.9

# Kernelparams
max_kernel_size = 1000



agent = Agent_Model(state_dim=6, lr=lr, momentum=momentum)

# TODO Hier muss noch eine Ladefunktion rein


matmul_kernel = load_kernel()  # Load Kernel



ema = 0
is_in_expl = False
# Lists
ema_list = []
flops_list_model = []
flops_list_expl = []
metrik_list_expl = []
metrik_list_model = []


for episode in range(episodes):

    
    N = np.random.randint(50, max_kernel_size)
    M = np.random.randint(50, max_kernel_size)
    P = np.random.randint(50, max_kernel_size)
    
    #N = 100
    #M = 200
    #P = 150
    
    FLOP = 2 * N * M * P
    TFLOP = FLOP / 1e+12
    
    elements = N * P
    
    # Generate Matrixes for later Kernel execution
    A, B, C = generate_matrixes(N, M, P)
    
    state = env_init(N, M, P, max_kernel_size)

    if (episode + 1) == 1:
        action = exploration()
        print("Eploration-setzen: ", action)

        action_next = cp.round((action + 1) ** 5)
        action_next = action_next.squeeze()

        tile_size, block_size = action_next

        tile_size = int(tile_size.get())
        block_size = int(block_size.get())

        in_first_expl = True

        
    # Exploration - Modellgegenüberstellung
    if (episode +1) %2 == 0 and (episode +1) != 2:
        action_expl = exploration()
        action_model = agent.forward(state)

        action_next_expl = cp.round((action_expl + 1) ** 5)      
        action_next_model = cp.round((action_model + 1) ** 5)

        action_next_expl = action_next_expl.squeeze()
        action_next_model = action_next_model.squeeze() 

        tile_size_expl, block_size_expl = action_next_expl
        tile_size_model, block_size_model = action_next_model  

        tile_size_expl = int(tile_size_expl.get())
        block_size_expl = int(block_size_expl.get())

        tile_size_model = int(tile_size_model.get())
        block_size_model = int(block_size_model.get())

        is_in_expl = True



    if is_in_expl != True and in_first_expl != True:
        # Modell berechnet action
        action = agent.forward(state)
        print("Modell-output: ", action)

        action_next = cp.round((action + 1) ** 5)
        action_next = action_next.squeeze()

        tile_size, block_size = action_next
        print("Tile-Size: ", tile_size, " - Block-Size: ", block_size)

        tile_size = int(tile_size.get())
        block_size = int(block_size.get())


    
    if is_in_expl != True:

        # =============================================================
        # being executet, if in_first_expl = True OR it is NOT an exploration-episode
        # =============================================================

        metrik, numbers_r_zero, TFLOPs = run(tile_size, block_size, N, M, P, A, B, C, FLOP, matmul_kernel, alpha_for_metrik)


        if in_first_expl == True:
            

            ema = (ema * beta_for_ema) + ((metrik * (3/4)) * (1-beta_for_ema))

        
        if in_first_expl != True:
            flops_list_model.append(TFLOPs)
            metrik_list_model.append(metrik)

            reward = metrik - ema
            ema = (ema * beta_for_ema) + (metrik * (1-beta_for_ema))
        
            delta_performance = reward * (-1)
        

            # =============================================================
            # !This is an additional exploration, since it's needed to compare Models prediction with exploration.
            # !But it's not needed for training.
            # =============================================================

            action_expl = exploration()
            action_next_expl = cp.round((action_expl + 1) ** 5)   
            action_next_expl = action_next_expl.squeeze()

            tile_size_expl, block_size_expl = action_next_expl
            tile_size_expl = int(tile_size_expl.get())
            block_size_expl = int(block_size_expl.get())

            metrik_expl, _, TFLOPs_expl = run(tile_size_expl, block_size_expl, N, M, P, A, B, C, FLOP, matmul_kernel, alpha_for_metrik)
            
            flops_list_expl.append(TFLOPs_expl)
            metrik_list_expl.append(metrik_expl)

        
             

    
    if is_in_expl == True:


        # ===========================================
        # Exploration_Kernel_Run
        # ===========================================


        metrik_expl, _, TFLOPs_expl = run(tile_size_expl, block_size_expl, N, M, P, A, B, C, FLOP, matmul_kernel, alpha_for_metrik)
        
        flops_list_expl.append(TFLOPs_expl)
        metrik_list_expl.append(metrik_expl)

        reward_expl = metrik_expl - ema


        # ===========================================
        # Model_Kernel_Run
        # ===========================================

        
        metrik_model, numbers_r_zero, TFLOPs_model = run(tile_size, block_size, N, M, P, A, B, C, FLOP, matmul_kernel, alpha_for_metrik)
        flops_list_model.append(TFLOPs_model)
        metrik_list_model.append(metrik_model)

        
        reward_model = metrik_model - ema

        ema = (ema * beta_for_ema) + (metrik * (1-beta_for_ema))

        delta_performance = reward_expl - reward_model



    if in_first_expl != True:
        agent.backward(delta_performance)
        
        
        
    
    numbers_r_zero = (C == 0).sum()     # You can use it, if you prefer to quickly check for any zeros!
    
        
    ema_list.append(ema)    
    if (episode +1) == 1:
        print("Exponential Moving Average: ", ema)    
        print("TFLOP/s at 1. Exploration: ", TFLOPs)
    if (episode + 1) != 1:    
        print(f"Episode: {episode + 1} - Zeros: {(numbers_r_zero/elements * 100):.4f}% - Delta_performance: {delta_performance:.4f} - Metrik: {metrik:.4f} - TFLOP: {(FLOP * 1e-12):.6f} - TFLOP/s: {(TFLOPs):.4f}")
        print("Exponential Moving Average: ", ema)
        
        print("-----------------------------------------------------------------------")
        
    # Reset Exploration-Flags
    in_first_expl = False
    is_in_expl = False

# Comparison between kernel results (GPU) and etablished results (CPU) - making sure we have a proper calculation!
print("Results -GPU:\n", C, "\n")

print("Results -CPU:\n", np.matmul(A, B), "\n")    
del C    

print(len(flops_list_model))



# you can leave it or delete it, doesen't matter, since it's use is for evaluation purposes only.
plt.figure(figsize=(8, 5))  
plt.plot(range(1, episodes ), metrik_list_model, label='metric-model', color='blue')
#plt.plot(range(1, episodes + 1), ema_list, label='EMA', color='red')
#plt.plot(range(1, episodes ), metrik_list_expl, label = 'metric-exploration', color='yellow')
plt.xlabel("Iteration")
plt.ylabel("metric")
plt.legend()
plt.show()
