"""
    Scheduler(schedules...)

Callback for hyperparameter scheduling.
Takes a pair of hyperparameters and schedules from ParameterSchedulers.

## Example
```julia
lrschedule = Exp(0.1, 0.5)
scheduler = Scheduler(
    LearningRate => lrschedule
)
```
"""
mutable struct Scheduler <: Callback
    schedules::Dict{Type{<:HyperParameter}, ParameterSchedulers.AbstractSchedule}
    step::Int
    Scheduler(args...; kwargs...) = new(Dict(args...; kwargs...), 1)
end

Base.show(io::IO, scheduler::Scheduler) =
    print(io, "Scheduler(", join(keys(scheduler.schedules), ", "), ")")

function FluxTraining.stateaccess(scheduler::Scheduler)
    # TODO: implement proper merging of permissions
    if length(keys(scheduler.schedules)) == 0
        hpstateaccess = (;)
    else
        hpstateaccess = merge(FluxTraining.stateaccess.(keys(scheduler.schedules))...)
    end
    return (data = Read(), cbstate = (; hyperparams = Write(), history = Read()),
            hpstateaccess...)
end


function FluxTraining.init!(scheduler::Scheduler, learner)
    if !haskey(learner.cbstate, :hyperparams)
        learner.cbstate.hyperparams = ValueHistories.MVHistory()
    end
    scheduler.step = 1

    return scheduler
end


function FluxTraining.on(::StepBegin, phase::AbstractTrainingPhase, scheduler::Scheduler, learner)
    for (H, schedule) in scheduler.schedules
        value = schedule(scheduler.step)
        FluxTraining.sethyperparameter!(learner, H, value)
        push!(
            learner.cbstate.hyperparams,
            Symbol(H),
            learner.cbstate.history[phase].steps,
            value)
    end
    scheduler.step += 1
end
