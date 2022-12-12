from ctypes import *
import numpy as np
import faulthandler

faulthandler.enable()

Abc_NtkType_t = c_int
Abc_NtkFunc_t = c_int
word = c_ulong  # platform specific

ABC_OBJ_NUMBER = 11

st__hash_func_type = CFUNCTYPE(c_int, c_char_p, c_int)
st__compare_func_type = CFUNCTYPE(c_int, c_char_p, c_char_p)
Abc_Frame_Callback_BmcFrameDone_Func = CFUNCTYPE(None, c_int, c_int, c_int)

### STRUCTURE DEFINITIONS ###
class Abc_Cex_t(Structure):
    pass

class Extra_MmFlex_t(Structure):
    _fields_ = [
        ('nEntriesUsed', c_int),
        ('pCurrent', c_char_p),
        ('pEnd', c_char_p),
        ('nChunkSize', c_int),
        ('nChunksAlloc', c_int),
        ('nChunks', c_int),
        ('pChuncks', POINTER(c_char_p)),
        ('nMemoryUsed', c_int),
        ('nMemoryAlloc', c_int)
    ]

class Mem_Fixed_t(Structure):
    _fields_ = [
        ('nEntrySize', c_int),
        ('nEntriesAlloc', c_int),
        ('nEntriesUsed', c_int),
        ('nEntriesMax', c_int),
        ('pEntriesFree', c_char_p),
        ('nChunkSize', c_int),
        ('nChunksAlloc', c_int),
        ('nChunks', c_int),
        ('pChunks', POINTER(c_char_p)),
        ('nMemoryUsed', c_int),
        ('nMemoryAlloc', c_int),
    ]

class Mem_Step_t(Structure):
    pass

Mem_Step_t._fields_ = [
        ('nMems', c_int),
        ('pMems', Mem_Fixed_t),
        ('nMapSize', c_int),
        ('pMap', Mem_Fixed_t),
        ('nLargeChunksAlloc', c_int),
        ('nLargeChunks', c_int),
        ('pLargeChunks', POINTER(POINTER(None))),
    ]

class Vec_Ptr_t(Structure):
    _fields_ = [
        ('nCap', c_int),
        ('nSize', c_int),
        ('pArray', POINTER(POINTER(None)))
    ]

class Vec_Int_t(Structure):
    _fields_ = [
        ('nCap', c_int),
        ('nSize', c_int),
        ('pArray', POINTER(c_int)),
    ]

class Vec_Str_t(Structure):
    _fields_ = [
        ('nCap', c_int),
        ('nSize', c_int),
        ('pArray', c_char_p)
    ]

class Vec_Wec_t(Structure):
    _fields_ = [
        ('nCap', c_int),
        ('nSize', c_int),
        ('pArray', POINTER(Vec_Int_t))
    ]

class Vec_Wrd_t(Structure):
    _fields_ = [
        ('nCap', c_int),
        ('nSize', c_int),
        ('pArray', POINTER(word))
    ]

class Vec_Vec_t(Structure):
    _fields_ = [
        ('nCap', c_int),
        ('nSize', c_int),
        ('pArray', POINTER(POINTER(None))),
    ]

class Vec_Flt_t(Structure):
    _fields_ = [
        ('nCap', c_int),
        ('nSize', c_int),
        ('pArray', POINTER(c_float)),
    ]

class Vec_Bit_t(Structure):
    _fields_ = [
        ('nCap', c_int),
        ('nSize', c_int),
        ('pArray', POINTER(c_int)),
    ]

class st__table_entry(Structure):
    pass

st__table_entry._fields_ = [
        ('key', c_char_p),
        ('record', c_char_p),
        ('next', POINTER(st__table_entry))
    ]

class st__table(Structure):
    _fields_ = [
        ('compare', st__compare_func_type),
        ('hash', st__hash_func_type),
        ('num_bins', c_int),
        ('num_entries', c_int),
        ('max_density', c_int),
        ('reorder_flag', c_int),
        ('grow_factor', c_double),
        ('bins', POINTER(POINTER(st__table_entry))),
    ]

class Abc_Des_t(Structure):
    pass

Abc_Des_t._fields_ = [
    ('pName', c_char_p),
    ('pManFunc', POINTER(None)),
    ('vTops', POINTER(Vec_Ptr_t)),
    ('vModules', POINTER(Vec_Ptr_t)),
    ('tModules', POINTER(st__table)),
    ('pLibrary', c_char_p),
    ('pGenlib', c_char_p),
]

class Nm_Entry_t(Structure):
    pass
    
Nm_Entry_t._fields_ = [
        ('Type', c_uint),                           # object type
        ('ObjId', c_uint),                          # object ID
        ('pNextI2N', POINTER(Nm_Entry_t)),          # the next entry in the ID hash table
        ('pNextN2I', POINTER(Nm_Entry_t)),          # the next entry in the name hash table
        ('pNameSake', POINTER(Nm_Entry_t)),         # the next entry with the same name
        ('Name', c_char * 0)                        # name of the object
    ]

class Nm_Man_t(Structure):
    _fields_ = [
        ('pBinsI2N', POINTER(POINTER(Nm_Entry_t))), # mapping IDs into names
        ('pBinsN2I', POINTER(POINTER(Nm_Entry_t))), # mapping names into IDs 
        ('nBins', c_int),                           # the number of bins in tables
        ('nEntries', c_int),                        # the number of entries
        ('nSizeFactor', c_int),                     # determined how much larger the table should be
        ('nGrowthFactor', c_int),                   # determined how much the table grows after resizing
        ('pMem', POINTER(Extra_MmFlex_t,))          # memory manager for entries (and names)
    ]

class FILE(Structure):
    pass


class Gia_Rpr_t(Structure):
    _fields_ = [
        ('iRepr', c_uint, 28),
        ('fProved', c_uint, 1),
        ('fFailed', c_uint, 1),
        ('fColorA', c_uint, 1),
        ('fColorB', c_uint, 1),
    ]

class Gia_Plc_t(Structure):
    _fields_ = [
        ('fFixed', c_uint, 1),
        ('xCoord', c_uint, 15),
        ('fUndef', c_uint, 1),
        ('yCoord', c_uint, 15),
    ]

class Gia_Obj_t(Structure):
    _fields_ = [
        ('iDiff0', c_uint, 29),
        ('fCompl0', c_uint, 1),
        ('fMark0', c_uint, 1),
        ('fTerm', c_uint, 1),
        ('iDiff1', c_uint, 29),
        ('fCompl1', c_uint, 1),
        ('fMark1', c_uint, 1),
        ('fPahse', c_uint, 1),
        ('Value', c_uint),
    ]

class Gia_Dat_t(Structure):
    pass

class Gia_Man_t(Structure):
    pass

Gia_Man_t._fields_ = [
    ('pName', c_char_p),                        # name of the AIG
    ('pSpec', c_char_p),                        # name of the input file
    ('nRegs', c_int),                           # number of registers
    ('nRegsAlloc', c_int),                      # number of allocated registers
    ('nObjs', c_int),                           # number of objects
    ('nObjsAlloc', c_int),                      # number of allocated objects
    ('pObjs', POINTER(Gia_Obj_t)),              # the array of objects
    ('pMuxes', POINTER(c_uint)),                # control signals of MUXes
    ('nXors', c_int),                           # the number of XORs
    ('nMuxes', c_int),                          # the number of MUXes 
    ('nBufs', c_int),                           # the number of buffers
    ('vCis', POINTER(Vec_Int_t)),               # the vector of CIs (PIs + LOs)
    ('vCos', POINTER(Vec_Int_t)),               # the vector of COs (POs + LIs)
    ('vHash',    Vec_Int_t),                    # hash links
    ('vHTable',  Vec_Int_t),                    # hash table
    ('fAddStrash', c_int),                      # performs additional structural hashing
    ('fSweeper',   c_int),                      # sweeper is running
    ('fGiaSimple', c_int),                      # simple mode (no const-propagation and strashing)
    ('vRefs', Vec_Int_t),                       # the reference count
    ('pRefs', POINTER(c_int)),                  # the reference count
    ('pLutRefs', POINTER(c_int)),               # the reference count
    ('vLevels',  Vec_Int_t),                    # levels of the nodes
    ('nLevels', POINTER(c_int)),                # the mamixum level
    ('nConstrs', POINTER(c_int)),               # the number of constraints
    ('nTravIds', POINTER(c_int)),               # the current traversal ID
    ('nFront', POINTER(c_int)),                 # frontier size 
    ('pReprsOld', POINTER(c_int)),              # representatives (for CIs and ANDs)
    ('pReprs',  POINTER(Gia_Rpr_t)),            # representatives (for CIs and ANDs)
    ('pNexts',  POINTER(c_int)),                # next nodes in the equivalence classes
    ('pSibls',  POINTER(c_int)),                # next nodes in the choice nodes
    ('pIso',    POINTER(c_int)),                # pairs of structurally isomorphic nodes
    ('nTerLoop',   c_int),                      # the state where loop begins  
    ('nTerStates', c_int),                      # the total number of ternary states
    ('pFanData',    POINTER(c_int)),            # the database to store fanout information
    ('nFansAlloc', c_int),                      # the size of fanout representation
    ('vFanoutNums',  POINTER(Vec_Int_t)),       # static fanout
    ('vFanout',      POINTER(Vec_Int_t)),       # static fanout
    ('vMapping',     POINTER(Vec_Int_t)),       # mapping for each node
    ('vMapping2',    POINTER(Vec_Int_t)),       # mapping for each node
    ('vFanouts2',    POINTER(Vec_Int_t)),       # mapping fanouts 
    ('vCellMapping', POINTER(Vec_Int_t)),       # mapping for each node
    ('pSatlutWinman', POINTER(None)),           # windowing for SAT-based mapping
    ('vPacking', POINTER(Vec_Int_t)),           # packing information
    ('vConfigs', POINTER(Vec_Int_t)),           # cell configurations
    ('pCellStr', c_char_p),                     # cell description
    ('vLutConfigs', POINTER(Vec_Int_t)),        # LUT configurations
    ('vEdgeDelay',  POINTER(Vec_Int_t)),        # special edge information
    ('vEdgeDelayR', POINTER(Vec_Int_t)),        # special edge information
    ('vEdge1',      POINTER(Vec_Int_t)),        # special edge information
    ('vEdge2',      POINTER(Vec_Int_t)),        # special edge information
    ('pCexComb', POINTER(Abc_Cex_t)),           # combinational counter-example
    ('pCexSeq', POINTER(Abc_Cex_t)),            # sequential counter-example
    ('vSeqModelVec', POINTER(Vec_Ptr_t)),       # sequential counter-examples
    ('vCopies', Vec_Int_t),                     # intermediate copies
    ('vCopies2', Vec_Int_t),                    # intermediate copies
    ('vVar2Obj', POINTER(Vec_Int_t)),           # mapping of variables into objects
    ('vTruths', POINTER(Vec_Int_t)),            # used for truth table computation
    ('vFlopClasses', POINTER(Vec_Int_t)),       # classes of flops for retiming/merging/etc
    ('vGateClasses', POINTER(Vec_Int_t)),       # classes of gates for abstraction
    ('vObjClasses', POINTER(Vec_Int_t)),        # classes of objects for abstraction
    ('vInitClasses', POINTER(Vec_Int_t)),       # classes of flops for retiming/merging/etc
    ('vRegClasses', POINTER(Vec_Int_t)),        # classes of registers for sequential synthesis
    ('vRegInits', POINTER(Vec_Int_t)),          # initial state
    ('vDoms', POINTER(Vec_Int_t)),              # dominators
    ('vBarBufs', POINTER(Vec_Int_t)),           # barrier buffers
    ('vXors', POINTER(Vec_Int_t)),              # temporary XORs
    ('pSwitching', POINTER(c_ubyte)),           # switching activity for each object
    ('pPlacement', POINTER(Gia_Plc_t)),         # placement of the objects
    ('pAigExtra', POINTER(Gia_Man_t)),          # combinational logic of holes
    ('vInArrs', POINTER(Vec_Flt_t)),            # PI arrival times
    ('vOutReqs', POINTER(Vec_Flt_t)),           # PO required times
    ('vCiArrs', POINTER(Vec_Int_t)),            # CI arrival times
    ('vCoReqs', POINTER(Vec_Int_t)),            # CO required times
    ('vCoArrs', POINTER(Vec_Int_t)),            # CO arrival times
    ('vCoAttrs', POINTER(Vec_Int_t)),           # CO attributes
    ('vWeights', POINTER(Vec_Int_t)),           # object attributes
    ('And2Delay', c_int),                       # delay of the AND gate 
    ('DefInArrs', c_float),                       # default PI arrival times
    ('DefOutReqs', c_float),                      # default PO required times
    ('vSwitching', POINTER(Vec_Int_t)),         # switching activity
    ('pTravIds', POINTER(c_int)),               # separate traversal ID representation
    ('nTravIdsAlloc', c_int),                   # the number of trav IDs allocated
    ('vNamesIn', POINTER(Vec_Ptr_t)),           # the input names 
    ('vNamesOut', POINTER(Vec_Ptr_t)),          # the output names
    ('vNamesNode', POINTER(Vec_Ptr_t)),         # the node names
    ('vUserPiIds', POINTER(Vec_Int_t)),         # numbers assigned to PIs by the user
    ('vUserPoIds', POINTER(Vec_Int_t)),         # numbers assigned to POs by the user
    ('vUserFfIds', POINTER(Vec_Int_t)),         # numbers assigned to FFs by the user
    ('vCiNumsOrig', POINTER(Vec_Int_t)),        # original CI names
    ('vCoNumsOrig', POINTER(Vec_Int_t)),        # original CO names
    ('vIdsOrig', POINTER(Vec_Int_t)),           # original object IDs
    ('vIdsEquiv', POINTER(Vec_Int_t)),          # original object IDs proved equivalent
    ('vCofVars', POINTER(Vec_Int_t)),           # cofactoring variables
    ('vClockDoms', POINTER(Vec_Vec_t)),         # clock domains
    ('vTiming', POINTER(Vec_Flt_t)),            # arrival/required/slack
    ('pManTime', POINTER(None)),                # the timing manager
    ('pLutLib', POINTER(None)),                 # LUT library
    ('nHashHit', word),                         # hash table hit
    ('nHashMiss', word),                        # hash table miss
    ('pData', POINTER(None)),                   # various user data
    ('pData2', POINTER(c_uint)),                # various user data
    ('iData', c_int),                           # various user data
    ('iData2', c_int),                          # various user data
    ('nAnd2Delay', c_int),                      # AND2 delay scaled to match delay numbers used
    ('fVerbose', c_int),                        # verbose reports
    ('MappedArea', c_int),                      # area after mapping
    ('MappedDelay', c_int),                     # delay after mapping
    # bit-parallel simulation
    ('fBuiltInSim', c_int),
    ('iPatsPi', c_int),
    ('nSimWords', c_int),
    ('nSimWordsT', c_int),
    ('iPastPiMax', c_int),
    ('nSimWordsMax', c_int),
    ('vSims', POINTER(Vec_Wrd_t)),
    ('vSimsT', POINTER(Vec_Wrd_t)),
    ('vSimsPi', POINTER(Vec_Wrd_t)),
    ('vSimsPo', POINTER(Vec_Wrd_t)),
    ('vClassOld', POINTER(Vec_Int_t)),
    ('vClassNew', POINTER(Vec_Int_t)),
    ('vPats', POINTER(Vec_Int_t)),
    ('vPolars', POINTER(Vec_Bit_t)),
    # incremental simulation
    ('fIncrSim', c_int),
    ('iNextPi', c_int),
    ('iTimeStamp', c_int),
    ('vTimeStamps', POINTER(Vec_Int_t)),
    # truth table computation for small functions
    ('nTtVars', c_int),                         # truth table variables
    ('nTtWords', c_int),                        # truth table words
    ('vTtNums', POINTER(Vec_Int_t)),            # object numbers
    ('vTtNodes', POINTER(Vec_Int_t)),           # internal nodes
    ('vTtInputs', POINTER(Vec_Ptr_t)),          # truth tables for constant and primary inputs
    ('vTtMemory', POINTER(Vec_Wrd_t)),          # truth tables for internal nodes
    # balancing
    ('vSuper', POINTER(Vec_Int_t)),             # supergate
    ('vStore', POINTER(Vec_Int_t)),             # node storage  
    # existential quantification
    ('iSuppPi', c_int),                         # the number of support variables
    ('nSuppWords', c_int),                      # the number of support words
    ('vSuppWords', POINTER(Vec_Wrd_t)),         # support information
    ('vCopiesTwo', Vec_Int_t),                  # intermediate copies
    ('vSuppVars', Vec_Int_t),                   # used variables
    ('vVarMap', Vec_Int_t),                     # used variables
    ('pUData', POINTER(Gia_Dat_t)),
]
    
class Abc_Time_t(Structure):
    _fields_ = [
        ('Rise', c_float),
        ('Fall', c_float)
    ]

class Abc_ManTime_t(Structure):
    _fields_ = [
        ('tArrDef', Abc_Time_t),
        ('tReqDef', Abc_Time_t),
        ('vArrs', POINTER(Vec_Ptr_t)),
        ('vReqs', POINTER(Vec_Ptr_t)),
        ('tInDriveDef', Abc_Time_t),
        ('tOutLoadDef', Abc_Time_t),
        ('tInDrive', POINTER(Abc_Time_t)),
        ('tOutLoad', POINTER(Abc_Time_t)),
    ]

Abc_Cex_t._fields_ = [
    ('iPo', c_int),
    ('iFrame', c_int),
    ('nRegs', c_int),
    ('nPis', c_int),
    ('nBits', c_int),
    ('pData', c_uint * 0)
]

class Abc_Nam_t(Structure):
    _fields_ = [
        # info storage for names
        ('nStore', c_int),
        ('iHandle', c_int),
        ('pStore', c_char_p),
        # internal number mappings
        ('vInt2Handle', Vec_Int_t),
        ('vInt2Next', Vec_Int_t),
        # hash table for names
        ('pBins', c_int),
        ('nBins', c_int),
        # manage recycling
        ('nRefs', c_int),
        # internal buffer
        ('vBuffer', Vec_Str_t)
    ]

class Abc_Ntk_t(Structure):
    pass

Abc_Ntk_t._fields_ = [
    ('ntkType', Abc_NtkType_t),
    ('ntkFunc', Abc_NtkFunc_t),
    ('pName', c_char_p),
    ('pSpec', c_char_p),
    ('pManName', POINTER(Nm_Man_t)),
    ('vObjs', POINTER(Vec_Ptr_t)),
    ('vPis', POINTER(Vec_Ptr_t)),
    ('vPos', POINTER(Vec_Ptr_t)),
    ('vCis', POINTER(Vec_Ptr_t)),
    ('vCos', POINTER(Vec_Ptr_t)),
    ('vPios', POINTER(Vec_Ptr_t)),
    ('vBoxes', POINTER(Vec_Ptr_t)),
    ('vLtlProperties', POINTER(Vec_Ptr_t)),
    ('nObjCounts', c_int * int(ABC_OBJ_NUMBER)),
    ('nObjs', c_int),
    ('nConstrs', c_int),
    ('nBarBufs', c_int),
    ('nBarBufs2', c_int),
    ('pNetBackup', POINTER(Abc_Ntk_t)),
    ('iStep', c_int),
    ('pDesign', POINTER(Abc_Des_t)),
    ('pAltView', POINTER(Abc_Ntk_t)),
    ('fHieVisited', c_int),
    ('fHiePath', c_int),
    ('Id', c_int),
    ('dTemp', c_double),
    ('nTravIds', c_int),
    ('vTravIds', Vec_Int_t),
    ('pMmObj', POINTER(Mem_Fixed_t)),
    ('pMmStep', POINTER(Mem_Step_t)),
    ('pManFunc', POINTER(None)),
    ('pManTime', POINTER(Abc_ManTime_t)),
    ('pManCut', POINTER(None)),
    ('AndGateDelay', c_float),
    ('LevelMax', c_int),
    ('vLevelsR', POINTER(Vec_Int_t)),
    ('vSupps', POINTER(Vec_Ptr_t)),
    ('pModel', POINTER(c_int)),
    ('pSeqModel', POINTER(Abc_Cex_t)),
    ('vSeqModelVec', POINTER(Vec_Ptr_t)),
    ('pExdc', POINTER(Abc_Ntk_t)),
    ('pExcare', POINTER(None)),
    ('pData', POINTER(None)),
    ('pCopy', POINTER(Abc_Ntk_t)),
    ('pBSMan', POINTER(None)),
    ('pSCLib', POINTER(None)),
    ('vGates', POINTER(Vec_Int_t)),
    ('vPhases', POINTER(Vec_Int_t)),
    ('pWLoadUsed', c_char_p),
    ('pLutTimes', POINTER(c_float)),
    ('vOnehots', POINTER(Vec_Ptr_t)),
    ('vObjPerm', POINTER(Vec_Int_t)),
    ('vTopo', POINTER(Vec_Int_t)),
    ('vAttrs', POINTER(Vec_Ptr_t)),
    ('vNameIds', POINTER(Vec_Int_t)),
    ('vFins', POINTER(Vec_Int_t)),
]

class Abc_Frame_t(Structure):
    pass

Abc_Frame_t._fields_ = [
    ('sVersion', c_char_p),
    ('sBinary', c_char_p),
    ('tCommands', POINTER(st__table)),
    ('tAliases', POINTER(st__table)),
    ('tFlags', POINTER(st__table)),
    ('aHistory', POINTER(Vec_Ptr_t)),
    ('pNtkCur', POINTER(Abc_Ntk_t)),
    ('pNtkBestDelay', POINTER(Abc_Ntk_t)),
    ('pNtkBestArea', POINTER(Abc_Ntk_t)),
    ('pNtkBackup', POINTER(Abc_Ntk_t)),
    ('nSteps', c_int),
    ('fSource', c_int),
    ('fAutoexac', c_int),
    ('fBatchMode', c_int),
    ('fBridgeMode', c_int),
    ('pNtkBest', POINTER(Abc_Ntk_t)),
    ('nBestNtkArea', c_float),
    ('nBestNtkDelay', c_float),
    ('nBestNtkNodes', c_int),
    ('nBestNtkLevels', c_int),
    ('Out', POINTER(FILE)),
    ('Err', POINTER(FILE)),
    ('Hst', POINTER(FILE)),
    ('TimeCommand', c_double),
    ('TimeTotal', c_double),
    ('vStore', POINTER(Vec_Ptr_t)),
    ('pManDec', POINTER(None)),
    ('pManDsd', POINTER(None)),
    ('pManDsd2', POINTER(None)),
    ('pLibLut', POINTER(None)),
    ('pLibBox', POINTER(None)),
    ('pLibGen', POINTER(None)),
    ('pLibGen2', POINTER(None)),
    ('pLibSuper', POINTER(None)),
    ('pLibScl', POINTER(None)),
    ('pAbcCon', POINTER(None)),
    ('pDrivingCell', c_char_p),
    ('MaxLoad', c_float),
    ('vIndFlops', POINTER(Vec_Int_t)),
    ('nIndFrames', c_int),
    ('pGia', POINTER(Gia_Man_t)),
    ('pGia2', POINTER(Gia_Man_t)),
    ('pGiaBest', POINTER(Gia_Man_t)),
    ('pGiaBest2', POINTER(Gia_Man_t)),
    ('pGiaSaved', POINTER(Gia_Man_t)),
    ('nBestLuts', c_int),
    ('nBestEdges', c_int),
    ('nBestLevels', c_int),
    ('nBestLuts2', c_int),
    ('nBestEdges2', c_int),
    ('nBestLevels2', c_int),
    ('pCex', POINTER(Abc_Cex_t)),
    ('pCex2', POINTER(Abc_Cex_t)),
    ('vCexVec', POINTER(Vec_Ptr_t)),
    ('vPoEquivs', POINTER(Vec_Ptr_t)),
    ('vStatuses', POINTER(Vec_Int_t)),
    ('vAbcObjIds', POINTER(Vec_Int_t)),
    ('Status', c_int),
    ('nFrames', c_int),
    ('vPlugInComBinPairs', POINTER(Vec_Ptr_t)),
    ('vLTLProperties_global', POINTER(Vec_Ptr_t)),
    ('vSignalNames', POINTER(Vec_Ptr_t)),
    ('pSpecName', c_char_p),
    ('pSave1', POINTER(None)),
    ('pSave2', POINTER(None)),
    ('pSave3', POINTER(None)),
    ('pSave4', POINTER(None)),
    ('pAbc85Ntl', POINTER(None)),
    ('pAbc85Ntl2', POINTER(None)),
    ('pAbc85Best', POINTER(None)),
    ('pAbc85Delay', POINTER(None)),
    ('pAbcWlc', POINTER(None)),
    ('pAbcWlcInv', POINTER(Vec_Int_t)),
    ('pAbcRtl', POINTER(None)),
    ('pAbcBac', POINTER(None)),
    ('pAbcCba', POINTER(None)),
    ('pAbcPla', POINTER(None)),
    ('pJsonStrs', POINTER(Abc_Nam_t)),
    ('vJsonObjs', POINTER(Vec_Wec_t)),
    ('pGiaMiniAig', POINTER(Gia_Man_t)),
    # ('ds', POINTER(DdManager)) # #ifdef ABC_USE_CUDD #endif
    ('pGiaMiniLut', POINTER(Gia_Man_t)),
    ('vCopyMiniAig', POINTER(Vec_Int_t)),
    ('vCopyMiniLut', POINTER(Vec_Int_t)),
    ('pArray', POINTER(c_int)),
    ('pBoxes', POINTER(c_int)),
    ('pNdr', POINTER(None)),
    ('pNdrArray', POINTER(c_int)),
    ('pFuncOnFrameDone', Abc_Frame_Callback_BmcFrameDone_Func),
]

class SC_Lib(Structure):
    pass

class SC_Man(Structure):
    pass
# LOAD THE LIBRARY

lib = cdll.LoadLibrary("extern/abc/libabc.so")

Abc_Start = getattr(lib, "Abc_Start")
Abc_Start.argtypes = []
Abc_Start.restype = None

Abc_Stop = getattr(lib, "Abc_Stop")
Abc_Stop.argtypes = []
Abc_Stop.restype = None

Abc_FrameGetGlobalFrame = getattr(lib, "Abc_FrameGetGlobalFrame")
Abc_FrameGetGlobalFrame.argtypes = []
Abc_FrameGetGlobalFrame.restype = POINTER(Abc_Frame_t)

Cmd_CommandIsDefined = getattr(lib, "Cmd_CommandIsDefined")
Cmd_CommandIsDefined.argtypes = [POINTER(Abc_Frame_t), c_char_p]
Cmd_CommandIsDefined.restype = c_int

Cmd_CommandExecute = getattr(lib, "Cmd_CommandExecute")
Cmd_CommandExecute.argtypes = [POINTER(Abc_Frame_t), c_char_p]
Cmd_CommandExecute.restype = c_int

Abc_FrameReadNtk = getattr(lib, "Abc_FrameReadNtk")
Abc_FrameReadNtk.argtypes = [POINTER(Abc_Frame_t)]
Abc_FrameReadNtk.restype = POINTER(Abc_Ntk_t)

Abc_RLfLOGetMaxDelayTotalArea = getattr(lib, "Abc_RLfLOGetMaxDelayTotalArea")
Abc_RLfLOGetMaxDelayTotalArea.argtypes = [POINTER(Abc_Frame_t), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int, c_int, c_int]
Abc_RLfLOGetMaxDelayTotalArea.restype = None

Abc_RLfLOGetNumNodesAndLevels = getattr(lib, "Abc_RLfLOGetNumNodesAndLevels")
Abc_RLfLOGetNumNodesAndLevels.argtypes = [POINTER(Abc_Frame_t), POINTER(c_int), POINTER(c_int)]
Abc_RLfLOGetNumNodesAndLevels.restype = None

Abc_RLfLOPrintNodeIds = getattr(lib, "Abc_RLfLOPrintNodeIds")
Abc_RLfLOPrintNodeIds.argtypes = [POINTER(Abc_Frame_t)]
Abc_RLfLOPrintNodeIds.restype = None

Abc_RLfLOGetObjTypes = getattr(lib, "Abc_RLfLOGetObjTypes")
Abc_RLfLOGetObjTypes.argtypes = [POINTER(Abc_Frame_t), np.ctypeslib.ndpointer(dtype=c_int, ndim=2, flags='C_CONTIGUOUS')]
Abc_RLfLOGetObjTypes.restype = None

Abc_RLfLOGetNumEdges = getattr(lib, "Abc_RLfLOGetNumEdges")
Abc_RLfLOGetNumEdges.argtypes = [POINTER(Abc_Frame_t), POINTER(c_int)]
Abc_RLfLOGetNumEdges.restype = None

Abc_RLfLOGetEdges = getattr(lib, "Abc_RLfLOGetEdges")
Abc_RLfLOGetEdges.argtypes = [POINTER(Abc_Frame_t), np.ctypeslib.ndpointer(dtype=c_int, ndim=2, flags='C_CONTIGUOUS'), c_int, np.ctypeslib.ndpointer(dtype=c_int, ndim=2, flags='C_CONTIGUOUS') ]
Abc_RLfLOGetEdges.restype = None

Abc_RLfLOGetNumObjs = getattr(lib, "Abc_RLfLOGetNumObjs")
Abc_RLfLOGetNumObjs.argtyoes = [POINTER(Abc_Frame_t), POINTER(c_int)]
Abc_RLfLOGetNumObjs.restype = None

Abc_RLfLOPrintObjNum2x = getattr(lib, "Abc_RLfLOPrintObjNum2x")
Abc_RLfLOPrintObjNum2x.argtypes = [POINTER(Abc_Frame_t)]
Abc_RLfLOPrintObjNum2x.restype = None

Abc_RLfLOSizeofInt = getattr(lib, "Abc_RLfLOSizeofInt")
Abc_RLfLOSizeofInt.argtypes = [POINTER(c_size_t)]
Abc_RLfLOSizeofInt.restype = None

Abc_RLfLOMapGetAreaDelay = getattr(lib, "Abc_RLfLOMapGetAreaDelay")
Abc_RLfLOMapGetAreaDelay.argtypes = [POINTER(Abc_Frame_t), POINTER(c_float), POINTER(c_float), c_int, c_int, c_double, c_int, c_int]
Abc_RLfLOMapGetAreaDelay.restype = c_int

def Abc_RLfLOGetObjTypes_wrapper(pAbc):
    num_objs = c_int()
    Abc_RLfLOGetNumObjs(pAbc, byref(num_objs))
    arr = np.ones( (num_objs.value, 1), dtype=c_int ) * (-10)
    Abc_RLfLOGetObjTypes(pAbc, arr)
    return arr

def Abc_RLfLOGetEdges_wrapper(pAbc):
    num_edges = c_int()
    Abc_RLfLOGetNumEdges(pAbc, byref(num_edges))
    edge_index = np.ones( (2, num_edges.value), dtype=c_int ) * (-5)
    edge_attr = np.ones( (num_edges.value, 1), dtype=c_int ) * (-6)
    Abc_RLfLOGetEdges(pAbc, edge_index, num_edges, edge_attr)
    return edge_index, edge_attr


if __name__ == "__main__":
    def Start_and_load():
        Abc_Start()
        pAbc = Abc_FrameGetGlobalFrame()
        lib_loaded = Cmd_CommandExecute(pAbc, b'read /home/kunzro/workspace/Reinforcement_Learning_for_Logic_Optimization/libraries/asap7.lib')
        ntk_loaded = Cmd_CommandExecute(pAbc, b'read /home/kunzro/workspace/Reinforcement_Learning_for_Logic_Optimization/circuits/adder.v')
        return pAbc


    def testMapAreaDelay():
        MaxDelay = c_float()
        TotalArea = c_float()
        NumNodes = c_int()
        NumLevels = c_int()
        MaxDelay2 = c_float()
        TotalArea2 = c_float()
        NumNodes2 = c_int()
        NumLevels2 = c_int()
        actions = [
            'rewrite',
            'balance',
            'rewrite',
            'resub',
            'rewrite -z',
            'balance',
            'balance',
            'refactor',
            'balance',
            'resub -z'
        ]
        results_old = []
        for i in range(len(actions)):
            # performe all the mapping in the "old" way
            pAbc = Start_and_load()
            for inner_action in actions[:i+1]:
                Cmd_CommandExecute(pAbc, b'strash')
                Cmd_CommandExecute(pAbc, inner_action.encode('utf-8'))
            Abc_RLfLOGetNumNodesAndLevels(pAbc, byref(NumNodes), byref(NumLevels))
            Cmd_CommandExecute(pAbc, b'map')
            Abc_RLfLOGetMaxDelayTotalArea(pAbc, byref(TotalArea), byref(MaxDelay), 0, 0, 0, 0, 0)
            Abc_Stop()
            results_old.append((MaxDelay.value, TotalArea.value, NumNodes.value, NumLevels.value))

        pAbc = Start_and_load()
        results_new = []
        for action in actions:
            Cmd_CommandExecute(pAbc, b'strash')
            Cmd_CommandExecute(pAbc, action.encode('utf-8'))
            Abc_RLfLOMapGetAreaDelay(pAbc, byref(MaxDelay2), byref(TotalArea2), 0, 0, 0, 0, 0)
            Abc_RLfLOGetNumNodesAndLevels(pAbc, byref(NumNodes2), byref(NumLevels2))
            results_new.append((MaxDelay2.value, TotalArea2.value, NumNodes2.value, NumLevels2.value))

        for res1, res2 in zip(results_new, results_old):
            for val1, val2 in zip(res1, res2):
                assert val1 == val2, "different values found!"
                print(f"val1: {val1} val2: {val2}")

    testMapAreaDelay()

    if False:
        Abc_Start()
        pAbc = Abc_FrameGetGlobalFrame()
        pAbc2 = Abc_FrameGetGlobalFrame()
        print(f"pAbc == pAbc2: {addressof(pAbc.contents) == addressof(pAbc2.contents)}, pAbc: {pAbc}, pAbc2: {pAbc2}")
        print(f"pAbc.contents == pAbc[0]: {addressof(pAbc.contents) == addressof(pAbc[0])}")
        read_def = Cmd_CommandIsDefined(pAbc, b'read')
        print(read_def)
        lib_loaded = Cmd_CommandExecute(pAbc, b'read /home/kunzro/workspace/Reinforcement_Learning_for_Logic_Optimization/libraries/asap7.lib')
        print(f'lib_loaded: {lib_loaded}; 0 = success, 1 = failed')
        ntk_loaded = Cmd_CommandExecute(pAbc, b'read /home/kunzro/workspace/Reinforcement_Learning_for_Logic_Optimization/circuits/adder.v')
        print(f'ntk_loaded: {ntk_loaded}; 0 = success, 1 = failed')


        Cmd_CommandExecute(pAbc, b'map')
        MaxDelay = c_float()
        TotalArea = c_float()
        NumNodes = c_int()
        NumLevels = c_int()
        Abc_RLfLOGetMaxDelayTotalArea(pAbc, byref(MaxDelay), byref(TotalArea), 0, 0, 0, 0, 0)
        Abc_RLfLOGetNumNodesAndLevels(pAbc, byref(NumNodes), byref(NumLevels))
        print(f"MaxDelay: {MaxDelay.value}, TotalArea: {TotalArea.value}, NumNodes: {NumNodes}, NumLevels: {NumLevels}")
        Cmd_CommandExecute(pAbc, b'stime')
        Cmd_CommandExecute(pAbc, b'print_stats')

        Cmd_CommandExecute(pAbc, b'strash')
        Cmd_CommandExecute(pAbc, b'rewrite')

        Cmd_CommandExecute(pAbc, b'map')
        Abc_RLfLOGetMaxDelayTotalArea(pAbc, byref(MaxDelay), byref(TotalArea), 0, 0, 0, 0, 0)
        Abc_RLfLOGetNumNodesAndLevels(pAbc, byref(NumNodes), byref(NumLevels))
        print(f"MaxDelay: {MaxDelay.value}, TotalArea: {TotalArea.value}, NumNodes: {NumNodes}, NumLevels: {NumLevels}")
        Cmd_CommandExecute(pAbc, b'stime')
        Cmd_CommandExecute(pAbc, b'print_stats')

        Cmd_CommandExecute(pAbc, b'strash')
        Cmd_CommandExecute(pAbc, b'rewrite')

        Cmd_CommandExecute(pAbc, b'map')
        Abc_RLfLOGetMaxDelayTotalArea(pAbc, byref(MaxDelay), byref(TotalArea), 0, 0, 0, 0, 0)
        Abc_RLfLOGetNumNodesAndLevels(pAbc, byref(NumNodes), byref(NumLevels))
        print(f"MaxDelay: {MaxDelay.value}, TotalArea: {TotalArea.value}, NumNodes: {NumNodes.value}, NumLevels: {NumLevels.value}")
        Cmd_CommandExecute(pAbc, b'stime')
        Cmd_CommandExecute(pAbc, b'print_stats')

        Cmd_CommandExecute(pAbc, b'strash')
        #Cmd_CommandExecute(pAbc, b'map')

        #Abc_RLfLOPrintNodeIds(pAbc)
        size = c_size_t()
        Abc_RLfLOSizeofInt(byref(size))
        print(f"The size of integers is: {size}")

        num_objs = c_int()
        Abc_RLfLOGetNumObjs(pAbc, byref(num_objs))
        print(f"the number of objects are: {num_objs}")

        num_edges = c_int()
        Abc_RLfLOGetNumEdges(pAbc, byref(num_edges))
        print(f"The number of edges are: {num_edges}")

        node_types = Abc_RLfLOGetObjTypes_wrapper(pAbc=pAbc)
        print(f"the node types are: {node_types}")

        edge_index, edge_attr = Abc_RLfLOGetEdges_wrapper(pAbc=pAbc)
        print(f"the edge_index are: {edge_index}")
        print(f"the edge_attr are: {edge_attr}")

        Abc_RLfLOPrintObjNum2x(pAbc)

        Abc_Stop()