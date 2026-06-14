from rdkit import Chem
from rdkit.Chem import Draw
import random
import pandas as pd
import numpy as np
from VirMolAnalyte.DataPrepare import ShiftPrediction,GetCarbonType,CtypeNumMW,Smi2PubchemFP



def random_connected_substructures(smiles, keep_ratio=0.5, n_samples=10, seed=None, img_size=(250,250)):
    """
    从分子中随机采样多个连通子结构，去重并返回 DataFrame 和绘图结果
    
    参数:
        smiles (str): 输入分子 SMILES
        keep_ratio (float): 保留原子比例 (0-1)
        n_samples (int): 采样次数
        seed (int): 随机种子
        img_size (tuple): 分子图片大小
    
    返回:
        df (pd.DataFrame): 子分子的 SMILES、保留原子编号、去掉原子编号
        img (PIL.Image): 原始分子 + 所有子分子的拼接图
    """
    if seed is not None:
        random.seed(seed)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("输入的 SMILES 无效")

    num_atoms = mol.GetNumAtoms()
    target_size = max(1, int(num_atoms * keep_ratio))

    results = []
    submols = [mol]
    legends = ["Original"]

    for i in range(n_samples):
        # Step 1: 随机起点
        start_atom = random.choice(range(num_atoms))
        selected = {start_atom}

        # Step 2: 扩展邻居直到达到目标
        while len(selected) < target_size:
            possible_neighbors = []
            for atom_idx in selected:
                atom = mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    if n_idx not in selected:
                        possible_neighbors.append(n_idx)
            if not possible_neighbors:
                break
            new_atom = random.choice(possible_neighbors)
            selected.add(new_atom)

        # Step 3: 构建子分子
        editable = Chem.EditableMol(mol)
        for idx in sorted(range(num_atoms), reverse=True):
            if idx not in selected:
                editable.RemoveAtom(idx)
        submol = editable.GetMol()
        sub_smiles = Chem.MolToSmiles(submol)

        kept_atoms = sorted(list(selected))
        removed_atoms = sorted([i for i in range(num_atoms) if i not in kept_atoms])

        results.append({
            "sampled_smiles": sub_smiles,
            "kept_atoms": kept_atoms,
            "removed_atoms": removed_atoms
        })

        submols.append(submol)
        legends.append(f"Sample {i+1}: {sub_smiles}")

    # 去重：按 sampled_smiles 去重
    df = pd.DataFrame(results).drop_duplicates(subset=["sampled_smiles"]).reset_index(drop=True)

    # 绘制原始分子 + 所有子分子
    img = Draw.MolsToGridImage(
        submols,
        legends=legends,
        molsPerRow=3,
        subImgSize=img_size
    )
    return df, img

def all_single_atom_deletions(smiles):
    """
    对分子进行单原子删除，返回所有删除一个原子的子分子 DataFrame
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("输入的 SMILES 无效")

    num_atoms = mol.GetNumAtoms()
    results = []

    for idx in range(num_atoms):
        editable = Chem.EditableMol(mol)
        editable.RemoveAtom(idx)
        try:
            submol = editable.GetMol()
            sub_smiles = Chem.MolToSmiles(submol)
            kept_atoms = sorted([i for i in range(num_atoms) if i != idx])
            removed_atoms = [idx]
            results.append({
                "sampled_smiles": sub_smiles,
                "kept_atoms": kept_atoms,
                "removed_atoms": removed_atoms
            })
        except:
            # 删除后不合法的分子，跳过
            continue

    df = pd.DataFrame(results).drop_duplicates(subset=["sampled_smiles"]).reset_index(drop=True)
    return df


def combined_substructures(smiles, keep_ratio=0.5, n_samples=10, seed=None, img_size=(250,250)):
    """
    合并随机连通子结构采样和单原子删除结果
    
    参数:
        smiles (str): 输入分子 SMILES
        keep_ratio (float): 随机子结构保留原子比例
        n_samples (int): 随机采样次数
        seed (int): 随机种子
        img_size (tuple): 分子图片大小
    
    返回:
        df (pd.DataFrame): 合并后的子分子信息
        img (PIL.Image): 原始分子 + 所有子分子的拼接图
    """
    # 调用第一个方法
    df_random, img_random = random_connected_substructures(smiles, keep_ratio, n_samples, seed, img_size)
    
    # 调用第二个方法
    df_single = all_single_atom_deletions(smiles)
    
    # 合并 DataFrame 并去重
    df_combined = pd.concat([df_random, df_single], ignore_index=True).drop_duplicates(subset=["sampled_smiles"]).reset_index(drop=True)
    
    # 为绘图准备所有分子
    mol = Chem.MolFromSmiles(smiles)
    mols = [mol]  # 原始分子
    legends = ["Original"]
    
    # 添加随机采样分子
    for smi in df_random["sampled_smiles"]:
        try:
            mols.append(Chem.MolFromSmiles(smi))
            legends.append(smi)
        except:
            continue
    
    # 添加单原子删除分子（避免重复）
    existing_smiles = set(df_random["sampled_smiles"])
    for smi in df_single["sampled_smiles"]:
        if smi not in existing_smiles:
            try:
                mols.append(Chem.MolFromSmiles(smi))
                legends.append(smi)
            except:
                continue
    
    # 绘制所有分子
    img = Draw.MolsToGridImage(
        mols,
        legends=legends,
        molsPerRow=3,
        subImgSize=img_size
    )
    
    return df_combined, img

def VirDBGenerator_V1(smiles,keep_ratio=0.5,n_samples=10,seed=None,img_size=(250,250),save_path="./results/generated_virDB.npz"):
    df_generated,img=combined_substructures(smiles, keep_ratio=keep_ratio, n_samples=n_samples, seed=seed, img_size=img_size)
    smileslist=df_generated["sampled_smiles"].tolist()
    df_generated.to_csv(save_path.replace(".npz",".csv"),index=False)

    print("_____chemical shift calculation_____")
    shiftlists,Normal_smileslist=ShiftPrediction(smileslist)
    CarbonType,CarbonNum=GetCarbonType(Normal_smileslist)
    
    print("_____PubchemFP calculation_____")
    Smi2PubchemFP(Normal_smileslist)
    df=pd.read_csv("descriptors.csv")
    df=df.drop("Name",axis=1)
    df1=df.dropna()
    normal=df1.index
    PuChemFP=np.array(df1)
    
    Normal_smileslist=[Normal_smileslist[i] for i in normal]
    shiftlists=[shiftlists[i] for i in normal]
    CarbonTypelist=[CarbonType[i] for i in normal]
    
    Ctype_Num_MW=CtypeNumMW(Normal_smileslist)

    kept_atoms=df_generated["kept_atoms"].tolist()

    kept_atoms=[kept_atoms[i] for i in normal]
    removed_atoms=df_generated["removed_atoms"].tolist()
    removed_atoms=[removed_atoms[i] for i in normal]

    DBindex=[i for i in range(len(shiftlists))]
    
    df2={}
    df2["smiles"]=Normal_smileslist
    df2["Vir_shifts"]=shiftlists
    df2["Ctype"]=CarbonTypelist
    df2["PubchemFP"]=PuChemFP
    df2["DBindex"]=DBindex
    df2["CtypeNum_and_MW"]= Ctype_Num_MW
    df2["kept_atoms"]=kept_atoms
    df2["removed_atoms"]=removed_atoms
    np.savez(save_path,data=df2)
    print("Job finish!")

