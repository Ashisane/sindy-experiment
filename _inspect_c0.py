import sys, os
sys.path.insert(0, r"C:\Users\UTKARSH\Desktop\mdg\c302")
sys.path.insert(0, r"C:\Users\UTKARSH\Desktop\mdg\mdg_build")

from c302 import parameters_C0
p = parameters_C0.ParameterisedModel()
print("TYPE:", type(p))
print("\nBIOPARAMS:")
for bp in p.bioparameters:
    print(f"  {bp.name!r:50s} = {bp.value!r}")
