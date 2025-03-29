Explanation

    Data Generation:
    The code simulates a ULA with 12 antennas. For each sample, a random source angle between -60° and 60° is chosen, and a corresponding steering vector is computed. Noise is added to simulate realistic signal conditions, and the real and imaginary parts are used as features.

    Model Architecture:
    The model comprises two branches:

        A CNN branch that uses Conv1D layers to extract local features from the antenna signals.

        A GCN branch that uses custom graph convolution (SimpleGCN) layers, utilizing a fixed adjacency matrix that reflects the ULA structure.

    The features from both branches are concatenated and passed through fully connected layers (MLP) to predict the DOA.

    Training and Evaluation:
    The model is trained with the Adam optimizer and early stopping is applied. Finally, the code evaluates the model using MAE and RMSE metrics, and plots the training history and the true vs. predicted angles.

This code should help simulate the results described in the paper and demonstrate the superiority of the proposed method over traditional approaches like MUSIC.
