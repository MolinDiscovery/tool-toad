from rdkit import Chem
import os
import numpy as np

__GAUXTB_CMD__ = "/groups/kemi/julius/opt/xtb_gaussian/xtb_external.py gbsa=methanol"

GAUSSIAN_COMMANDS = {
    "opt+freq": "opt freq b3lyp/6-31+g(d,p) scrf=(smd,solvent=methanol) empiricaldispersion=gd3 int=ultrafine",
    "ts_opt+freq": "opt=(ts,calcfc,noeigentest) freq b3lyp/6-31+g(d,p) scrf=(smd,solvent=methanol) empiricaldispersion=gd3 int=ultrafine",
    "irc_forward": "ircmax=(forward,readfc,maxpoints=10,recalc=3) b3lyp/6-31+g(d,p) scrf=(smd,solvent=methanol) empiricaldispersion=gd3 int=ultrafine",
    "irc_reverse": "ircmax=(reverse,readfc,maxpoints=10,recalc=3) b3lyp/6-31+g(d,p) scrf=(smd,solvent=methanol) empiricaldispersion=gd3 int=ultrafine",
    "ts_opt_xtb": "opt=(ts,calcall,noeigentest,nomicro) external={__GAUXTB_CMD__}",
    "opt_xtb": f"opt external={__GAUXTB_CMD__}",
    "freq_xtb": f"freq external={__GAUXTB_CMD__}",
}


def write_gaussian_input_file(
    mol_or_chk, name, command=GAUSSIAN_COMMANDS["opt+freq"], dir=".", mem=4, cpus=4
):
    """Writes .com file (Gaussian input file), taking structure from rdkit.Mol object or .chk file

    Args:
        mol_or_chk (rdkit.Mol or str): rdkit.Mol object or path to .chk file
        name (str): name of .com file
        command (str): route card for Gaussian calculation. Defaults to GAUSSIAN_COMMANDS["opt+freq"].
        dir (str, optional): Directory in which .com file will be written to. Defaults to ".".
        mem (int, optional): Link0 command. Defaults to 4.
        cpus (int, optional): Link0 command. Defaults to 4.

    Returns:
        str: Path to .com file
    """

    file_name = name + ".com"
    chk_file = name + ".chk"
    link0 = f"%mem={mem}\n%nprocshared={cpus}\n%chk={chk_file}\n"
    route = f"# {command}\n\n{name}\n\n"

    if type(mol_or_chk) == Chem.rdchem.Mol:
        mol = mol_or_chk
        charge = Chem.GetFormalCharge(mol)
        multiplicity = 1
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        conformers = mol.GetConformers()
        if len(conformers) > 1:
            raise Warning(f"{Chem.MolToSmiles(mol_or_chk)} has multiple conformers.")
        if not conformers:
            raise Exception("Mol is not embedded.")
        if not conformers[0].Is3D:
            raise Exception("Mol is not 3D.")
        route += f"{charge} {multiplicity}\n"
        coords = ""
        for atom, symbol in enumerate(symbols):
            p = conformers[0].GetAtomPosition(atom)
            line = " ".join((" " + symbol, str(p.x), str(p.y), str(p.z), "\n"))
            coords += line

    elif type(mol_or_chk) == str and mol_or_chk[-4:] == ".chk":
        link0 += f"%oldchk={mol_or_chk}\n"
        command += " Geom=AllCheck"
        coords = ""
        route = f"# {command}\n\n{name}\n\n"

    else:
        raise Exception(f"{mol_or_chk} is not a valid input.")

    with open(os.path.join(dir, file_name), "w") as file:
        file.write(link0 + route + coords + f"\n")

    return os.path.join(dir, file_name)


def get_gaussian_energy(out_file, e_type="scf"):
    """Reads Energy from Gaussian .out/.log file

    Args:
        out_file (str): Path to .out/.log file
        e_type (str, optional): Which Energy is returned ['scf' or 'gibbs']. Defaults to "scf".

    Returns:
        float: Energy in Hartree.
    """
    with open(out_file, "r") as ofile:
        line = ofile.readline()
        while line:
            if e_type == "scf":
                if (
                    "Recovered energy=" in line
                ):  # this is for externally calcualted energy
                    energy = float(line.split("=")[1].split(" ")[1])
                if "SCF Done:" in line:  # this is for internally calculated energy
                    energy = float(line.split(":")[1].split("=")[1].split("A.U.")[0])
            elif e_type == "gibbs":
                if "Sum of electronic and thermal Free Energies=" in line:
                    energy = float(line.split()[-1])
            else:
                raise Exception(
                    f"{e_type} is not a valid option out of ['scf', 'gibbs']."
                )
            line = ofile.readline()
    return energy
