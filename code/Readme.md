# Reconstructing the STRING PPI Input

## 1. Download the raw STRING network

STRING version: **v12.5**  
Organism: ***Populus trichocarpa***  
NCBI taxonomy ID: **3694**

Download:

https://stringdb-downloads.org/download/protein.links.v12.5/3694.protein.links.v12.5.txt.gz

The downloaded file is:

```text
3694.protein.links.v12.5.txt.gz
```

with columns:

```text
protein1    protein2    combined_score
```

## 2. Extract the STRING protein accessions

Run:

```bash
python extract_string_accessions.py   --input 3694.protein.links.v12.5.txt.gz   --output string_protein_accessions.txt
```

The script collects all unique identifiers appearing in `protein1` and `protein2` and removes the `3694.` taxonomy prefix.

The output contains one UniProt accession per line and is used as the input for UniProt ID Mapping.

## 3. Perform UniProt ID Mapping

Open the UniProt ID Mapping service:

https://www.uniprot.org/id-mapping

Use:

```text
From database: UniProtKB AC/ID
To database:   UniProtKB
```

Upload:

```text
string_protein_accessions.txt
```

After the mapping job is complete, download the result in `.xlsx`, `.tsv`, or `.csv` format.

The downloaded table must contain at least:

```text
From
Gene Names
```

## 4. Convert STRING protein identifiers to POPTR gene identifiers

Run:

```bash
python map_string_to_poptr.py   --string 3694.protein.links.v12.5.txt.gz   --mapping idmapping.xlsx   --output PPI.csv
```
