import os
import sys 
import tqdm

def processar(insertion_chance):
    if not os.path.exists('./alignedpt-'+str(insertion_chance)):
        os.mkdir('./alignedpt-'+str(insertion_chance))

    for file in tqdm.tqdm(os.listdir('./ocrpt-'+str(insertion_chance)+'/')):
        #print(file)
        cmd = 'python ochre/char_align.py ocrpt-'+str(insertion_chance)+'/'+file+' gspt-'+str(insertion_chance)+'/'+file+' mdpt-'+str(insertion_chance)+'/md'+file[:-4]+'.json --out_dir alignedpt-'+str(insertion_chance)+'/'
        #print(cmd)
        os.system(cmd)
    print('processado corretamente')


if __name__ == "__main__":
    insertion_chance = sys.argv[1]
    processar(insertion_chance)
