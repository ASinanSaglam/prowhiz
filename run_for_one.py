# Run from the prowhiz root                                                                                                                      
import sys; sys.path.insert(0, 'src')                     
from prowhiz.data.cif_parser import parse_cif                                                                                                    
from prowhiz.data.contacts import compute_contacts, CONTACT_TYPE_PAIRS

pdb_id = "1F8B"   # swap to any structure you want to check                                                                                      
sd = parse_cif(f"data/raw/{pdb_id}.cif", pdb_id=pdb_id, query_rcsb=False)                                                                        
# import IPython;IPython.embed()
print(f"Ligand: {sd.selection_info.selected.comp_id}  |  n_heavy={len(sd.ligand_atoms)}  |  n_protein={len(sd.protein_atoms)} | sel method={sd.selection_info.selection_method}")                  
                                                                                                                                                   
cl = compute_contacts(sd.protein_atoms, sd.ligand_atoms, cutoff=10.5)                                                                            
print(f"\nTotal contacts: {cl.num_contacts}")                                                                                                    
print(f"Contacts in allowed-element pairs (counted): {int(cl.contact_counts.sum())}")
print()                                                                                                                                          
for (a, b), count in zip(CONTACT_TYPE_PAIRS, cl.contact_counts):                                                                                 
    if count > 0:                                               
        print(f"  {a}-{b}: {int(count)}")   
