# Flow properties
flow_cadence    = 10
flow_property   = "(k*u + m*w)/omega"
flow_name       = 'Lin_Criterion'
flow            = flow_tools.GlobalFlowProperty(solver, cadence=flow_cadence)
flow.add_property(flow_property, name=flow_name)

# Logger parameters
time_factor     = T
endtime_str     = 'Sim end period: %f'
logger_cadence  = 100
iteration_str   = 'Iteration: %i, t/T: %e, dt/T: %e'
flow_log_message= 'Max linear criterion = {0:f}'

# Main loop
try:
    logger.info(endtime_str %(solver.stop_sim_time/time_factor))
    logger.info('Starting loop')
    start_time = time.time()
    dt = sbp.dt
    while solver.proceed:
        solver.step(dt)
        if solver.iteration % 1 == 0:
            w.set_scales(1)
            w_list.append(np.copy(w['g']))
            t_list.append(solver.sim_time)
        if solver.iteration % logger_cadence == 0:
            logger.info(iteration_str %(solver.iteration, solver.sim_time/time_factor, dt/time_factor))
            logger.info(flow_log_message.format(flow.max(flow_name)))
            if np.isnan(flow.max(flow_name)):
                raise NameError('Code blew up it seems')
