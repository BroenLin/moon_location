import os


def renamefordir(dir):
    files = os.listdir(dir)
    for file in files:
        name = os.path.join(dir, file)
        tail=file.split(".")[0].split("_")[-1]
        print(tail)
        newtail=tail.zfill(5)
        newfile=file.replace(tail,newtail)
        newname = os.path.join(dir, newfile)
        print(name," ",newname)
        os.rename(name, newname)

if __name__=="__main__":
    dir=r"D:\locate\moonlocate\data\ce4\descentimgs\all"
    renamefordir(dir)

    # dir="./data/ce4/baseimgs/NAC_DTM_CHANGE4_M1303619844_140CM_split4096"
    # files=os.listdir(dir)
    # for file in files:
    #     # print(file) ce4split_00_1.jpg
    #     name=os.path.join(dir,file)
    #
    #     print()
    #     head,r0,c0=file.split("_")
    #     c=c0.split(".")[0]
    #     print(r0,c)
    #     r=r0.zfill(3)
    #     c=c.zfill(3)
    #     print(r, c)
    #     new_name=head+"_"+r+"_"+c+".jpg"
    #     new_name=os.path.join(dir,new_name)
    #     # new_name=name.replace(r0,r).replace(c0,c)
    #
    #
    #     # new_name=name.replace("ce4split","ce4split2048")
    #     print(new_name)
    #     os.rename(name,new_name)