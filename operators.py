import bempp.api 
import numpy as np

def rescale(A, d1, d2):
    """Rescale the 2x2 block operator matrix A"""
    
    A[0, 1] = A[0, 1] * (d2 / d1)
    A[1, 0] = A[1, 0] * (d1 / d2)
    
    return A

def get_memory(number_of_scatterers, operator):
    mem = 0
    
    for i in range(2*number_of_scatterers):
        for j in range(2*number_of_scatterers):
            mem += operator[i,j].memory
    
    return mem/2 #returning half the memory as caching is used when an operator is used multiple times.

def PMCHWT_operator(grids, k_ext, k_int, mu_ext, mu_int, block_discretisation = False, preconditioner = False, 
                    parameters = None, type_of_preconditioner = False):
    """ Set up the PMCHWT operator/preconditioner """
    
    if preconditioner == False and type_of_preconditioner != False:
        raise ValueError("Type of preconditioner should only be defined when preconditioner == True")
    if preconditioner == True and type_of_preconditioner == False:
        type_of_preconditioner = 'diagonal'
        
    number_of_scatterers = len(grids)
    interior_operators = []
    exterior_operators = []
    identity_operators = []
    interior_electric_operators = []
    exterior_electric_operators = []
       
    #Setting up spaces. 
    #If preconditioner==True and block_discretisation==False then we use BC and RBC functions. 
    #If preconditioner==False and and block_discretisation==False we use RWG and SNC functions.
    #If block_discretisation is True then default discretisation is used
    if preconditioner == True and block_discretisation == False:
        domain_space = [bempp.api.function_space(grid, "BC", 0) for grid in grids]
        range_space = [bempp.api.function_space(grid, "BC", 0) for grid in grids]
        dual_to_range_space = [bempp.api.function_space(grid, "RBC", 0) for grid in grids]
    elif preconditioner == False and block_discretisation == False:
        domain_space = [bempp.api.function_space(grid, "RWG", 0) for grid in grids]
        range_space = [bempp.api.function_space(grid, "RWG", 0) for grid in grids]
        dual_to_range_space = [bempp.api.function_space(grid, "SNC", 0) for grid in grids]
                        
    for i in range(number_of_scatterers):
        if block_discretisation == True:
            A_ext = bempp.api.operators.boundary.maxwell.multitrace_operator(grids[i], k_ext, parameters = parameters)
            A_int = bempp.api.operators.boundary.maxwell.multitrace_operator(grids[i], k_int[i], parameters = parameters)
            
            ident = bempp.api.operators.boundary.sparse.multitrace_identity(grids[i], spaces='maxwell')                     
        else:
            A_ext = bempp.api.assembly.BlockedOperator(2,2)
            A_int = bempp.api.assembly.BlockedOperator(2,2)

            magnetic_field_ext = bempp.api.operators.boundary.maxwell.magnetic_field(domain_space[i], range_space[i],
                                                                                dual_to_range_space[i], k_ext,
                                                                                parameters = parameters)
            electric_field_ext = bempp.api.operators.boundary.maxwell.electric_field(domain_space[i], range_space[i], 
                                                                                dual_to_range_space[i], k_ext,
                                                                                parameters = parameters)
            A_ext[0,0] = magnetic_field_ext
            A_ext[0,1] = electric_field_ext
            A_ext[1,0] = -1 * electric_field_ext
            A_ext[1,1] = magnetic_field_ext

            magnetic_field_int = bempp.api.operators.boundary.maxwell.magnetic_field(domain_space[i], range_space[i],
                                                                                dual_to_range_space[i], k_int[i],
                                                                                parameters = parameters)
            electric_field_int = bempp.api.operators.boundary.maxwell.electric_field(domain_space[i], range_space[i], 
                                                                                dual_to_range_space[i], k_int[i],
                                                                                parameters = parameters)
            A_int[0,0] = magnetic_field_int
            A_int[0,1] = electric_field_int
            A_int[1,0] = -1 * electric_field_int
            A_int[1,1] = magnetic_field_int
            
            ident = bempp.api.assembly.BlockedOperator(2,2)
            identity = bempp.api.operators.boundary.sparse.identity(domain_space[i], range_space[i], dual_to_range_space[i])
            ident[0,0] = identity
            ident[1,1] = identity
        
        E_int = bempp.api.assembly.BlockedOperator(2,2)
        E_ext = bempp.api.assembly.BlockedOperator(2,2)
        
        A_ext = rescale(A_ext, k_ext, mu_ext)
        A_int = rescale(A_int, k_int[i], mu_int[i])
        
        E_int[0,1] = A_int[0,1]
        E_int[1,0] = A_int[1,0]
        
        E_ext[0,1] = A_ext[0,1]
        E_ext[1,0] = A_ext[1,0]
        
        interior_operators.append(A_int)
        exterior_operators.append(A_ext)
        identity_operators.append(ident)
        interior_electric_operators.append(E_int)
        exterior_electric_operators.append(E_ext)
       
    filter_operators = number_of_scatterers * [None]
    transfer_operators = np.empty((number_of_scatterers, number_of_scatterers), dtype=np.object)

    PMCHWT_op = bempp.api.BlockedOperator(2 * number_of_scatterers, 2 * number_of_scatterers)
    PMCHWT_pre = bempp.api.BlockedOperator(2 * number_of_scatterers, 2 * number_of_scatterers)
    
    for i in range(number_of_scatterers):
        filter_operators[i] = .5 * identity_operators[i] - interior_operators[i]
        for j in range(number_of_scatterers):
#             print(i,j)
            if i == j:
                # Create the diagonal elements
                if preconditioner == False:
                    element = interior_operators[j] + exterior_operators[j]
                elif type_of_preconditioner == 'diagonal':
                    element = interior_operators[j] + exterior_operators[j]
                elif type_of_preconditioner == 'exterior':
                    element = exterior_operators[j]
                elif type_of_preconditioner == 'interior':
                    element = interior_operators[j]
                elif type_of_preconditioner == 'interior_electric':
                    element = interior_electric_operators[j]
                elif type_of_preconditioner == 'exterior_electric':
                    element = exterior_electric_operators[j]
                else:
                    raise ValueError('Type of preconditioner can be False, diagonal, exterior, interior, interior_electric, exterior_electric')
            elif preconditioner == False:
                # Do the off-diagonal elements
                if block_discretisation == True:
                    Aij = bempp.api.operators.boundary.maxwell.multitrace_operator(grids[j], k_ext, target=grids[i], 
                                                                                   parameters = parameters)
                else:
                    Aij = bempp.api.BlockedOperator(2,2) 
                    magnetic_field = bempp.api.operators.boundary.maxwell.magnetic_field(domain_space[j], range_space[i], 
                                                                                         dual_to_range_space[i], k_ext, 
                                                                                         parameters=parameters) 
                    electric_field = bempp.api.operators.boundary.maxwell.electric_field(domain_space[j], range_space[i], 
                                                                                         dual_to_range_space[i], k_ext, 
                                                                                         parameters=parameters) 
                    
                    Aij[0,0] = magnetic_field
                    Aij[0,1] = electric_field
                    Aij[1,0] = -1 * electric_field
                    Aij[1,1] = magnetic_field

                element= rescale(Aij, k_ext, mu_ext)

            #Assign the 2x2 element to the block operator matrix.  
            if preconditioner == False:
                PMCHWT_op[2 * i, 2 * j] = element[0, 0]
                PMCHWT_op[2 * i, 2 * j + 1] = element[0, 1]
                PMCHWT_op[2 * i + 1, 2 * j] = element[1, 0]
                PMCHWT_op[2 * i + 1, 2 * j + 1] = element[1, 1] 
        if preconditioner == True:
            PMCHWT_pre[2*i, 2*i] = element[0, 0]
            PMCHWT_pre[2*i, 2*i + 1] = element[0, 1]
            PMCHWT_pre[2*i + 1, 2*i] = element[1, 0]
            PMCHWT_pre[2*i + 1, 2*i + 1] = element[1, 1]
            
    if preconditioner == True:
        return PMCHWT_pre
    else:
        return [PMCHWT_op, filter_operators]