import os
import pandas as pd
import numpy as np
for model_name in ["facebook/opt-350m"]:
    for split in ["valid", "test"]:
        for method in ["random_structured", "mask_gradient_l2_structured", "magnitude_l2_structured","ours_ppl_2", "ppl_only", "bias_only"]:
        # "ppl_only", "bias_only", "magnitude_l1_structured", "magnitude_l2_structured", "magnitude_linf_structured", "random_structured", "mask_gradient_l2_structured", 
        # for method in ["None"]:
        # for intraprocessing_method in ["random_perturbation","temperature_scaling"]:
            for gamma in ["0.01", "0.03","0.05","0.07","0.08","0.09"]:
            # for gamma in ["0.05", "0.1", "0.2","0.3","0.4","0.5","0.6", "0.7", "0.8"]:
                for seed in ["1", "2", "3"]:
                    for prompting in ["holistic"]:
                        # for pruned_heads_ratio in ["0"]:
                        for pruned_heads_ratio in np.linspace(0,0.2,11,endpoint=True):
                        #, "race_ethnicity", "religion", "sexual_orientation", "nationality"
                        #,"nationality", "race_ethnicity", "religion", "sexual_orientation"
                            for group in ["gender_and_sex"]:
                                alpha = "0.5"
                                # gamma = "0.5"                            
                                for head_knockout in ["None"]:
                                # for head_knockout in range(0,384):
                                    csv_directory = (
                                        "/scratch/abdel1/BOLD_2/ours/seed_"
                                        + str(seed)
                                        + "/output/"
                                        + "prompt_"
                                        + str(prompting)
                                        + "_h" + str(head_knockout)   
                                        + "_" + str(split)  
                                        + "/" + str(method) + "_" + str(pruned_heads_ratio) + "_alpha" + str(alpha) + "_gamma" + str(gamma) + "/"                    
                                    ) 
                                    
                                    file_name = (
                                        csv_directory
                                        + model_name.replace("/", "_")
                                        + "_"
                                        + str(group)
                                        + "_1_fixed" 
                                        + ".csv"
                                    )     
                                    print(file_name)  
                                    
                                    if os.path.exists(file_name):
                                        all_chunks_exist = True
                                        df = pd.read_csv(file_name)  
                                        for chunk in ["2", "3", "4", "5"]:
                                            if os.path.exists(file_name.replace("_1_fixed.csv", "_" + chunk + "_fixed.csv")):
                                                df_new = pd.read_csv(file_name.replace("_1_fixed.csv", "_" + chunk + "_fixed.csv"))
                                                df = pd.concat([df, df_new],axis=0,ignore_index=True,)
                                            else:
                                                print("yalahwy")
                                                all_chunks_exist = False
                                                break
                                            print(csv_directory)
                                        if all_chunks_exist:
                                            df.to_csv(file_name.replace("_1_fixed.csv", "_all_fixed.csv"),index=False)      


