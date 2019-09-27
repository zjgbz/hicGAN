import os, time, pickle, sys, math,random
import numpy as np
import hickle as hkl

def hic_matrix_extraction(DPATH, chrom_list, res=10000,norm_method='NONE'):
    hr_contacts_dict={}
    lr_contacts_dict={}
    for each in chrom_list:
        hr_hic_file = '../binary_data/float32/contact_matrix/MY_113.MY_115/MY_113.MY_115.%d.npy'%each
        lr_hic_file = '../binary_data/float32/contact_matrix/MY_113.MY_115_simulation1_seed1558030855_frac125/MY_113.MY_115_simulation1_seed1558030855_frac125.%d.npy'%each
        hr_contact_matrix = np.load(hr_hic_file)
        lr_contact_matrix = np.load(lr_hic_file)
        hr_contacts_dict['chr%d'%each] = hr_contact_matrix
        lr_contacts_dict['chr%d'%each] = lr_contact_matrix

    nb_hr_contacts={item:sum(sum(hr_contacts_dict[item])) for item in hr_contacts_dict.keys()}
    nb_lr_contacts={item:sum(sum(lr_contacts_dict[item])) for item in lr_contacts_dict.keys()}
    
    return hr_contacts_dict,lr_contacts_dict,nb_hr_contacts,nb_lr_contacts

def crop_hic_matrix_by_chrom(chrom,norm_type=0,size=40 ,thred=200):
    #thred=2M/resolution
    #norm_type=0-->raw count
    #norm_type=1-->log transformation
    #norm_type=2-->scaled to[-1,1]after log transformation, default
    #norm_type=3-->scaled to[0,1]after log transformation
    distance=[]
    crop_mats_hr=[]
    crop_mats_lr=[]    
    row,col = hr_contacts_norm_dict[chrom].shape
    if row<=thred or col<=thred:
        print('HiC matrix size wrong!')
        sys.exit()
    def quality_control(mat,thred=0.05):
        if len(mat.nonzero()[0])<thred*mat.shape[0]*mat.shape[1]:
            return False
        else:
            return True
        
    for idx1 in range(0,row-size,size):
        for idx2 in range(0,col-size,size):
            if abs(idx1-idx2)<thred:
                if quality_control(lr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]):
                    distance.append([idx1-idx2,chrom])
                    if norm_type==0:
                        lr_contact = lr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                        hr_contact = hr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                    elif norm_type==1:
                        lr_contact = lr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                        hr_contact = hr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                    elif norm_type==2:
                        lr_contact_norm = lr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                        hr_contact_norm = hr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                        lr_contact = lr_contact_norm*2.0/max_lr_contact_norm[chrom]-1
                        hr_contact = hr_contact_norm*2.0/max_hr_contact_norm[chrom]-1
                    elif norm_type==3:
                        lr_contact_norm = lr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                        hr_contact_norm = hr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                        lr_contact = lr_contact_norm*1.0/max_lr_contact_norm[chrom]
                        hr_contact = hr_contact_norm*1.0/max_hr_contact_norm[chrom]
                    else:
                        print('Normalization wrong!')
                        sys.exit()
                    
                    crop_mats_lr.append(lr_contact)
                    crop_mats_hr.append(hr_contact)
    crop_mats_hr = np.concatenate([item[np.newaxis,:] for item in crop_mats_hr],axis=0)
    crop_mats_lr = np.concatenate([item[np.newaxis,:] for item in crop_mats_lr],axis=0)
    return crop_mats_hr,crop_mats_lr,distance

def data_split(chrom_list,norm_type):
    random.seed(100)
    distance_all=[]
    assert len(chrom_list)>0
    hr_mats,lr_mats=[],[]
    for chrom in chrom_list:
        crop_mats_hr,crop_mats_lr,distance = crop_hic_matrix_by_chrom(chrom,norm_type,size=40 ,thred=200)
        distance_all+=distance
        hr_mats.append(crop_mats_hr)
        lr_mats.append(crop_mats_lr)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_mats=hr_mats.transpose((0,2,3,1))
    lr_mats=lr_mats.transpose((0,2,3,1))
    return hr_mats,lr_mats,distance_all

def main():
    cell=sys.argv[1]
    test_chr_start = int(sys.argv[2])
    test_size = int(sys.argv[3])
    test_chr_list = list(range(test_chr_start, test_chr_start + test_size))
    train_chr_list = list(range(1, 20))
    for test_chr in test_chr_list:
        train_chr_list.remove(test_chr)
    

    nb_hr_contacts_dir_filename = 'data/%s/nb_hr_contacts.hkl'%cell
    nb_lr_contacts_dir_filename = 'data/%s/nb_lr_contacts.hkl'%cell
    max_hr_contact_norm_dir_filename = 'data/%s/max_hr_contact_norm.hkl'%cell
    max_lr_contact_norm_dir_filename = 'data/%s/max_lr_contact_norm.hkl'%cell
    chrom_list = list(range(1, 20))
    if (not os.path.exists(nb_hr_contacts_dir_filename)) or (not os.path.exists(nb_lr_contacts_dir_filename)) or (not os.path.exists(max_hr_contact_norm_dir_filename)) or (not os.path.exists(max_lr_contact_norm_dir_filename)):
        # data extraction
        hr_contacts_dict,lr_contacts_dict,nb_hr_contacts,nb_lr_contacts = hic_matrix_extraction(cell, chrom_list)
        print("extraction completed.")

        max_hr_contact = max([nb_hr_contacts[item] for item in nb_hr_contacts.keys()])
        max_lr_contact = max([nb_lr_contacts[item] for item in nb_lr_contacts.keys()])

        #normalization
        hr_contacts_norm_dict = {item:np.log2(hr_contacts_dict[item]*max_hr_contact/sum(sum(hr_contacts_dict[item]))+1) for item in hr_contacts_dict.keys()}
        lr_contacts_norm_dict = {item:np.log2(lr_contacts_dict[item]*max_lr_contact/sum(sum(lr_contacts_dict[item]))+1) for item in lr_contacts_dict.keys()}

        max_hr_contact_norm={item:hr_contacts_norm_dict[item].max() for item in hr_contacts_dict.keys()}
        max_lr_contact_norm={item:lr_contacts_norm_dict[item].max() for item in lr_contacts_dict.keys()}

        hkl.dump(nb_hr_contacts,'data/%s/nb_hr_contacts.hkl'%cell)
        hkl.dump(nb_lr_contacts,'data/%s/nb_lr_contacts.hkl'%cell)

        hkl.dump(max_hr_contact_norm,'data/%s/max_hr_contact_norm.hkl'%cell)
        hkl.dump(max_lr_contact_norm,'data/%s/max_lr_contact_norm.hkl'%cell)
        print("save completed.")

    #hr_mats_train,lr_mats_train,distance_train = data_split(['chr%d'%idx for idx in list(range(1,16))],norm_type=2)
    #hr_mats_test,lr_mats_test,distance_test = data_split(['chr%d'%idx for idx in list(range(16,20))],norm_type=2)
    hr_mats_train,lr_mats_train,distance_train = data_split(['chr%d'%idx for idx in train_chr_list],norm_type=2)
    hr_mats_test,lr_mats_test,distance_test = data_split(['chr%d'%idx for idx in test_chr_list],norm_type=2)
    print("split completed.")
    hkl.dump([lr_mats_train,hr_mats_train],'data/%s/train_data_%d-%d.hkl'%(cell, test_chr_start, test_size))
    hkl.dump([lr_mats_test,hr_mats_test,distance_test],'data/%s/test_data_%d-%d.hkl'%(cell, test_chr_start, test_size))

    #hr_mats_train,lr_mats_train,distance_train = data_split(['chr%d'%idx for idx in list(range(19,20))],norm_type=2)
    #hr_mats_test,lr_mats_test,distance_test = data_split(['chr%d'%idx for idx in list(range(19,20))],norm_type=2)
    #hkl.dump([lr_mats_train,hr_mats_train],'data/%s/train_data.hkl'%cell)
    #hkl.dump([lr_mats_test,hr_mats_test],'data/%s/test_data.hkl'%cell)

if __name__ == '__main__':
	main()