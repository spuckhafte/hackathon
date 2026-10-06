"""Hand-built, source-verified IPC/CrPC/Evidence-Act -> BNS/BNSS/BSA table.

Kept in its own module so that `nyaya.statutes` stays readable: this file is
*data*, and every block names the source it was checked against.  Nothing here
was written from memory.

VERIFICATION
------------
Every mapping below was read out of the official Government of India
correspondence tables (prepared by the Central Academy for Police Training,
Bhopal - a BPR&D institution), fetched and text-extracted:

  IPC  -> BNS  : https://www.keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BNS_to_IPC.pdf
  CrPC -> BNSS : https://www.keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BNSS_to_CrPC.pdf
  IEA  -> BSA  : https://keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BSA_to_IEA.pdf

The official tables are printed new->old; they were inverted here, and each
inversion was cross-checked against the subject line that travels with the row
(the row giving BNS 103 <-> IPC 302 is headed "Punishment for murder"), which
makes a row misalignment detectable.

COMMENCEMENT.  The three Sanhitas came into force on **1 July 2024**, per
Gazette notification S.O. 850(E), MHA, 23 February 2024 (BNS; companion
S.O. 848(E)/849(E) for BNSS/BSA), which commences the BNS "except the provision
of sub-section (2) of section 106".  Copy of the gazette page:
https://www.livelaw.in/pdf_upload/bns-524228.pdf
Offences committed before that date are still tried under the old codes, so both
directions must stay queryable indefinitely - which is exactly why this is a
*query-time synonym class* and not an index-time rewrite.

SOURCES DELIBERATELY REJECTED (recorded so nobody re-introduces them):
  * uppolice.gov.in ".../IPC vis-a-vis BNS.pdf" is the Project 39A comparison of
    the **August 2023 Bill** (Bill 121 of 2023), which was withdrawn on
    12 Dec 2023.  Its clause numbers differ from the enacted Act (murder is
    clause 101 there, s.103 in the Act).
  * manupatracademy.com "Highlighting-important-sections..." extracts with digit
    errors (shows 116<->304A and 118<->306 where the Act has 106 and 108).
  * indiacode.nic.in returns HTTP 403 to automated fetches, so it could not be
    used as an independent diff.  Spot corroboration from SCC Online / LiveLaw
    agreed on every item checked.

NOTE ON SUB-SECTIONS.  `statutes.normalise_section` drops sub-clauses, so
"BNS 103(1)" canonicalises to ``bns§103``.  That is the right granularity for
retrieval (a judgment citing 103(1) and one citing 103 are the same provision),
but it means several old sections can collapse onto one new token - the reverse
crosswalk therefore legitimately returns lists.

Filled by Sumedha (M3).
"""
from __future__ import annotations

from typing import Dict, List

SOURCES: List[str] = [
    "https://www.keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BNS_to_IPC.pdf",
    "https://www.keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BNSS_to_CrPC.pdf",
    "https://keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BSA_to_IEA.pdf",
    "https://www.livelaw.in/pdf_upload/bns-524228.pdf  (S.O. 850(E), in force 1 July 2024)",
]

# ==========================================================================
# IPC, 1860  ->  Bharatiya Nyaya Sanhita, 2023          [source: BNS<->IPC table]
# ==========================================================================
# The official entry is given in a comment where the new provision is a
# sub-section, because the canonical token drops the sub-section.
_IPC_BNS: Dict[str, List[str]] = {
    # --- general principles ---
    "ipc§34": ["bns§3"],            # BNS 3(5) - common intention
    "ipc§120a": ["bns§61"],         # BNS 61(1) - criminal conspiracy defined
    "ipc§120b": ["bns§61"],         # BNS 61(2) - punishment for conspiracy
    "ipc§149": ["bns§190"],         # unlawful assembly, vicarious liability
    "ipc§511": ["bns§62"],          # attempt
    # --- homicide ---
    "ipc§299": ["bns§100"],         # culpable homicide
    "ipc§300": ["bns§101"],         # murder defined
    "ipc§302": ["bns§103"],         # BNS 103(1) - punishment for murder
    "ipc§303": ["bns§104"],
    "ipc§304": ["bns§105"],         # culpable homicide not amounting to murder
    "ipc§304a": ["bns§106"],        # BNS 106(1) - death by negligence
    "ipc§304b": ["bns§80"],         # dowry death
    "ipc§306": ["bns§108"],         # abetment of suicide
    "ipc§307": ["bns§109"],         # attempt to murder
    # --- hurt ---
    "ipc§319": ["bns§114"],
    "ipc§320": ["bns§116"],
    "ipc§321": ["bns§115"],         # BNS 115(1)
    "ipc§322": ["bns§117"],         # BNS 117(1)
    "ipc§323": ["bns§115"],         # BNS 115(2) - voluntarily causing hurt
    "ipc§324": ["bns§118"],         # BNS 118(1)
    "ipc§325": ["bns§117"],         # BNS 117(2)
    "ipc§326": ["bns§118"],         # BNS 118(2)
    # --- wrongful restraint / confinement ---
    "ipc§339": ["bns§126"],
    "ipc§340": ["bns§127"],
    "ipc§341": ["bns§126"],         # BNS 126(2)
    "ipc§342": ["bns§127"],         # BNS 127(2)
    # --- offences against women ---
    "ipc§354": ["bns§74"],          # outraging modesty
    "ipc§354a": ["bns§75"],         # sexual harassment
    "ipc§354c": ["bns§77"],         # voyeurism
    "ipc§354d": ["bns§78"],         # stalking
    "ipc§375": ["bns§63"],          # rape defined
    "ipc§376": ["bns§64", "bns§65"],  # 376(1),(2)->BNS 64 ; 376(3)->BNS 65(1)
    "ipc§376a": ["bns§66"],
    "ipc§376d": ["bns§70"],         # BNS 70(1) - gang rape
    "ipc§509": ["bns§79"],          # word/gesture insulting modesty
    # --- kidnapping ---
    "ipc§359": ["bns§137"],         # BNS 137(1)
    "ipc§361": ["bns§137"],         # BNS 137(1)
    "ipc§362": ["bns§138"],
    "ipc§363": ["bns§137"],         # BNS 137(2) - punishment for kidnapping
    # --- theft, robbery, dacoity ---
    "ipc§378": ["bns§303"],         # BNS 303(1) - theft defined
    "ipc§379": ["bns§303"],         # BNS 303(2) - punishment for theft
    "ipc§380": ["bns§305"],
    "ipc§390": ["bns§309"],         # BNS 309(1) - robbery defined
    "ipc§391": ["bns§310"],         # BNS 310(1) - dacoity defined
    "ipc§392": ["bns§309"],         # BNS 309(4) - punishment for robbery
    "ipc§395": ["bns§310"],         # BNS 310(2) - punishment for dacoity
    # --- criminal breach of trust (five IPC sections -> one BNS section) ---
    "ipc§405": ["bns§316"],         # BNS 316(1)
    "ipc§406": ["bns§316"],         # BNS 316(2)
    "ipc§407": ["bns§316"],         # BNS 316(3)
    "ipc§408": ["bns§316"],         # BNS 316(4)
    "ipc§409": ["bns§316"],         # BNS 316(5)
    # --- cheating ---
    "ipc§415": ["bns§318"],         # BNS 318(1) - cheating defined
    "ipc§417": ["bns§318"],         # BNS 318(2)
    "ipc§420": ["bns§318"],         # BNS 318(4) - cheating and dishonest inducement
    # --- forgery ---
    "ipc§463": ["bns§336"],         # BNS 336(1)
    "ipc§464": ["bns§335"],
    "ipc§465": ["bns§336"],         # BNS 336(2) - punishment for forgery
    "ipc§466": ["bns§337"],
    "ipc§467": ["bns§338"],
    "ipc§468": ["bns§336"],         # BNS 336(3) - forgery for cheating
    "ipc§471": ["bns§340"],         # BNS 340(2) - using a forged document
    # --- marriage / cruelty ---
    "ipc§493": ["bns§81"],
    "ipc§494": ["bns§82"],          # BNS 82(1) - bigamy
    "ipc§495": ["bns§82"],          # BNS 82(2)
    "ipc§498a": ["bns§85", "bns§86"],   # SPLIT: 85 offence, 86 defines "cruelty"
    # --- defamation, intimidation ---
    "ipc§499": ["bns§356"],         # BNS 356(1) - defamation defined
    "ipc§500": ["bns§356"],         # BNS 356(2) - punishment
    "ipc§503": ["bns§351"],         # BNS 351(1) - criminal intimidation
    "ipc§506": ["bns§351"],         # BNS 351(2)/(3)
    "ipc§507": ["bns§351"],         # BNS 351(4)
}

# ==========================================================================
# CrPC, 1973  ->  Bharatiya Nagarik Suraksha Sanhita, 2023  [source: BNSS table]
# ==========================================================================
_CRPC_BNSS: Dict[str, List[str]] = {
    "crpc§41": ["bnss§35"],         # CrPC 41 and 41A merged into BNSS 35
    "crpc§41a": ["bnss§35"],
    "crpc§41b": ["bnss§36"],
    "crpc§41c": ["bnss§37"],
    "crpc§41d": ["bnss§38"],
    "crpc§154": ["bnss§173"],       # FIR
    "crpc§161": ["bnss§180"],       # police statements
    "crpc§164": ["bnss§183"],       # magisterial recording of confessions
    "crpc§173": ["bnss§193"],       # charge sheet / final report
    "crpc§190": ["bnss§210"],       # cognizance
    "crpc§197": ["bnss§218"],       # prosecution of public servants
    "crpc§200": ["bnss§223"],       # complaint examination
    "crpc§300": ["bnss§337"],       # autrefois acquit / convict
    "crpc§313": ["bnss§351"],       # examination of the accused
    "crpc§357": ["bnss§395"],       # compensation
    "crpc§378": ["bnss§419"],       # appeal against acquittal
    "crpc§397": ["bnss§438"],       # revision - calling for records
    "crpc§400": ["bnss§441"],
    "crpc§401": ["bnss§442"],       # High Court's revisional powers
    "crpc§436": ["bnss§478"],       # bail in bailable offences
    "crpc§437": ["bnss§480"],       # bail in non-bailable offences
    "crpc§438": ["bnss§482"],       # anticipatory bail
    "crpc§439": ["bnss§483"],       # special powers re bail
    "crpc§482": ["bnss§528"],       # inherent powers of the High Court
}

# ==========================================================================
# Indian Evidence Act, 1872  ->  Bharatiya Sakshya Adhiniyam, 2023 [BSA table]
# ==========================================================================
_EVID_BSA: Dict[str, List[str]] = {
    "evid§3": ["bsa§2"],            # interpretation clause -> BSA 2(1)(a)..(l)
    "evid§5": ["bsa§3"],
    "evid§6": ["bsa§4"],            # res gestae
    "evid§7": ["bsa§5"],
    "evid§8": ["bsa§6"],            # motive, preparation, conduct
    "evid§23": ["bsa§21"],
    "evid§24": ["bsa§22"],          # BSA 22(1) - confession by inducement
    "evid§25": ["bsa§23"],          # BSA 23(1) - confession to a police officer
    "evid§26": ["bsa§23"],          # BSA 23(2)
    "evid§27": ["bsa§23"],          # proviso to BSA 23 - no number of its own
    "evid§28": ["bsa§22"],          # first proviso to BSA 22
    "evid§29": ["bsa§22"],          # second proviso to BSA 22
    "evid§30": ["bsa§24"],
    "evid§31": ["bsa§25"],
    "evid§32": ["bsa§26"],          # dying declaration
    "evid§33": ["bsa§27"],
    "evid§45": ["bsa§39"],          # BSA 39(1) - expert opinion
    "evid§65a": ["bsa§62"],
    "evid§65b": ["bsa§63"],         # electronic records certificate
    "evid§101": ["bsa§104"],        # burden of proof
    "evid§102": ["bsa§105"],
    "evid§103": ["bsa§106"],
    "evid§105": ["bsa§108"],
    "evid§106": ["bsa§109"],        # facts especially within knowledge
    "evid§114": ["bsa§119"],        # presumptions
    "evid§114a": ["bsa§120"],
}

CROSSWALK: Dict[str, List[str]] = {}
CROSSWALK.update(_IPC_BNS)
CROSSWALK.update(_CRPC_BNSS)
CROSSWALK.update(_EVID_BSA)

# ==========================================================================
# Where the mapping is NOT one-to-one, say so.  These notes are surfaced in the
# "why this result" panel whenever an expansion fired on one of them, because a
# user who searched 498A deserves to know the expansion covered two sections.
# ==========================================================================
CROSSWALK_NOTES: Dict[str, str] = {
    "ipc§34": "IPC 34 became BNS 3(5), a sub-section of the general-explanations "
              "section; 'BNS 34' is a different provision (private defence).",
    "ipc§120a": "IPC 120A and 120B were merged into BNS 61 (61(1) and 61(2)).",
    "ipc§120b": "IPC 120A and 120B were merged into BNS 61 (61(1) and 61(2)).",
    "ipc§302": "IPC 302 is BNS 103(1); BNS 103(2) is a new mob-lynching offence "
               "with no IPC counterpart.",
    "ipc§304a": "IPC 304A is BNS 106(1); BNS 106(2) (hit-and-run) is new and was "
                "expressly excluded from the 1 July 2024 commencement.",
    "ipc§321": "IPC 321 and 323 were merged into BNS 115 (115(1), 115(2)).",
    "ipc§323": "IPC 321 and 323 were merged into BNS 115 (115(1), 115(2)).",
    "ipc§322": "IPC 322 and 325 were merged into BNS 117; BNS 117(3)-(4) are new.",
    "ipc§325": "IPC 322 and 325 were merged into BNS 117; BNS 117(3)-(4) are new.",
    "ipc§324": "IPC 324 and 326 were merged into BNS 118 (118(1), 118(2)).",
    "ipc§326": "IPC 324 and 326 were merged into BNS 118 (118(1), 118(2)).",
    "ipc§359": "IPC 359, 361 and 363 all land in BNS 137 (359 and 361 both in 137(1)).",
    "ipc§361": "IPC 359, 361 and 363 all land in BNS 137 (359 and 361 both in 137(1)).",
    "ipc§363": "IPC 359, 361 and 363 all land in BNS 137; 363 is 137(2).",
    "ipc§376": "IPC 376 is split: 376(1)-(2) -> BNS 64, 376(3) -> BNS 65(1).",
    "ipc§378": "IPC 378 and 379 were merged into BNS 303 (303(1), 303(2)).",
    "ipc§379": "IPC 378 and 379 were merged into BNS 303 (303(1), 303(2)).",
    "ipc§390": "The robbery cluster IPC 390/392/393 collapsed into BNS 309.",
    "ipc§392": "The robbery cluster IPC 390/392/393 collapsed into BNS 309 (392 = 309(4)).",
    "ipc§391": "The dacoity cluster IPC 391/395 collapsed into BNS 310.",
    "ipc§395": "The dacoity cluster IPC 391/395 collapsed into BNS 310 (395 = 310(2)).",
    "ipc§405": "Five IPC sections (405-409) were merged into BNS 316(1)-(5); "
               "a bare 'BNS 316' citation is ambiguous.",
    "ipc§406": "Five IPC sections (405-409) were merged into BNS 316(1)-(5); "
               "406 is 316(2).",
    "ipc§407": "Five IPC sections (405-409) were merged into BNS 316(1)-(5).",
    "ipc§408": "Five IPC sections (405-409) were merged into BNS 316(1)-(5).",
    "ipc§409": "Five IPC sections (405-409) were merged into BNS 316(1)-(5); "
               "409 is 316(5).",
    "ipc§415": "IPC 415/417/420 were merged into BNS 318 (318(1), 318(2), 318(4)).",
    "ipc§417": "IPC 415/417/420 were merged into BNS 318.",
    "ipc§420": "IPC 415/417/420 were merged into BNS 318; 420 is 318(4).",
    "ipc§463": "IPC 463/465/468 were merged into BNS 336 (336(1), 336(2), 336(3)).",
    "ipc§465": "IPC 463/465/468 were merged into BNS 336; 465 is 336(2).",
    "ipc§468": "IPC 463/465/468 were merged into BNS 336; 468 is 336(3).",
    "ipc§471": "IPC 471 became BNS 340(2), a sub-section.",
    "ipc§493": "IPC 493/494/495 were renumbered into BNS 81 and BNS 82(1)-(2).",
    "ipc§494": "IPC 494 is BNS 82(1); IPC 495 is BNS 82(2).",
    "ipc§495": "IPC 494 is BNS 82(1); IPC 495 is BNS 82(2).",
    "ipc§498a": "NOT one-to-one: IPC 498A was SPLIT into BNS 85 (the offence) and "
                "BNS 86 (the definition of 'cruelty', formerly the Explanation). "
                "A 498A query must expand to both.",
    "ipc§499": "IPC 499 and 500 were merged into BNS 356 (356(1), 356(2)).",
    "ipc§500": "IPC 499 and 500 were merged into BNS 356; 500 is 356(2).",
    "ipc§503": "IPC 503/506/507 were merged into BNS 351; IPC 506 maps to BOTH "
               "351(2) and 351(3).",
    "ipc§506": "IPC 506 maps to BOTH BNS 351(2) and 351(3) (its two paragraphs).",
    "ipc§507": "IPC 503/506/507 were merged into BNS 351; 507 is 351(4).",
    "crpc§41": "CrPC 41 and 41A were merged into BNSS 35; BNSS 35(7) is new.",
    "crpc§41a": "CrPC 41 and 41A were merged into BNSS 35; BNSS 35(7) is new.",
    "crpc§482": "NUMBER COLLISION: CrPC 482 (inherent powers) became BNSS 528, "
                "while BNSS 482 is anticipatory bail (old CrPC 438). '482' means "
                "two different things depending on the code.",
    "crpc§438": "NUMBER COLLISION: CrPC 438 (anticipatory bail) became BNSS 482, "
                "and BNSS 438 is the old CrPC 397 (revision).",
    "crpc§397": "CrPC 397 became BNSS 438 - the same number 438 that used to mean "
                "anticipatory bail under the CrPC.",
    "evid§3": "The single interpretation clause (IEA 3) was exploded into the "
              "lettered definitions BSA 2(1)(a)-(l).",
    "evid§24": "IEA 24, 28 and 29 collapsed into BSA 22 plus its two provisos.",
    "evid§28": "IEA 28 is the first proviso to BSA 22, not a numbered section.",
    "evid§29": "IEA 29 is the second proviso to BSA 22, not a numbered section.",
    "evid§25": "IEA 25, 26 and 27 collapsed into BSA 23 (23(1), 23(2), proviso).",
    "evid§26": "IEA 25, 26 and 27 collapsed into BSA 23; 26 is 23(2).",
    "evid§27": "IEA 27 (the 'discovery' provision) has NO section number of its "
               "own in the BSA - it survives only as the proviso to BSA 23.",
    "evid§45": "IEA 45 became BSA 39(1).",
    "evid§65b": "Electronic records moved from IEA 65A/65B (secondary evidence) to "
                "BSA 62/63, where they are treated as primary evidence.",
    "evid§65a": "Electronic records moved from IEA 65A/65B to BSA 62/63.",
}

# ==========================================================================
# Omitted outright - there is nothing to expand to, and saying so is useful.
# ==========================================================================
CROSSWALK_OMITTED: Dict[str, str] = {
    "ipc§377": "IPC 377 (unnatural offences) was omitted from the BNS; there is no "
               "corresponding provision.  The official table states the s.377 "
               "reference was removed because 'the section 377 IPC has been "
               "excluded altogether'.",
    "ipc§497": "IPC 497 (adultery) was omitted from the BNS; there is no "
               "corresponding provision.  It appears nowhere in the official "
               "correspondence table.",
}
