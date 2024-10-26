import streamlit as st
import pyNN.neuron as sim
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate firing rate
def calculate_firing_rate(spike_data, runtime, bin_size=1.0):
    time_bins = np.arange(0, runtime + bin_size, bin_size)  # Create bins
    firing_rate = np.zeros(len(time_bins) - 1)
    for spiketrain in spike_data:
        firing_rate += np.histogram(spiketrain, bins=time_bins)[0]
    firing_rate = firing_rate / (len(spike_data) * (bin_size / 1000.0))  # Convert bin size to seconds
    return time_bins[:-1], firing_rate  # Return bin centers and firing rates

# Function to run the simulation
def run_simulation(input_spike_times, num_excitatory, num_inhibitory, synapse_type,
                   weight_exc, weight_inh, delay_exc, delay_inh, timestep, runtime,
                   cell_params, num_recorded, connector_params):
    sim.setup(timestep=timestep)

    # Create neuronal populations
    input_population = sim.Population(
        len(input_spike_times),
        sim.SpikeSourceArray(spike_times=input_spike_times),
        label="Input"
    )
    exc_neurons = sim.Population(
        num_excitatory,
        sim.IF_cond_exp(**cell_params),
        label="Excitatory"
    )
    inh_neurons = sim.Population(
        num_inhibitory,
        sim.IF_cond_exp(**cell_params),
        label="Inhibitory"
    )

    # Define synaptic connections with valid connectors
    try:
        if synapse_type == 'AllToAll':
            connector = sim.AllToAllConnector(
                allow_self_connections=connector_params.get('allow_self_connections', False)
            )
        elif synapse_type == 'OneToOne':
            connector = sim.OneToOneConnector()
        elif synapse_type == 'PairwiseBernoulli':
            connector = sim.FixedProbabilityConnector(
                p_connect=connector_params.get('p_connect', 0.5),
                allow_self_connections=connector_params.get('allow_self_connections', False)
            )
        elif synapse_type == 'FixedTotalNumber':
            connector = sim.FixedTotalNumberConnector(
                n=connector_params.get('n', 5),
                allow_self_connections=connector_params.get('allow_self_connections', False)
            )
        elif synapse_type == 'FixedInDegree':
            connector = sim.FixedNumberPostConnector(
                n=connector_params.get('n', 10),
                allow_self_connections=connector_params.get('allow_self_connections', False)
                # 'allow_multiple_connections' is NOT supported
            )
        elif synapse_type == 'FixedOutDegree':
            connector = sim.FixedNumberPreConnector(
                n=connector_params.get('n', 10),
                allow_self_connections=connector_params.get('allow_self_connections', False)
                # 'allow_multiple_connections' is NOT supported
            )
        else:
            st.error(f"Unknown synapse type: {synapse_type}")
            sim.end()
            return None, None, None
    except Exception as e:
        st.error(f"Error creating connector: {e}")
        sim.end()
        return None, None, None

    # Create synapses
    synapse_exc = sim.StaticSynapse(weight=weight_exc, delay=delay_exc)
    synapse_inh = sim.StaticSynapse(weight=weight_inh, delay=delay_inh)

    # Create projections
    input_to_exc = sim.Projection(input_population, exc_neurons, connector, synapse_exc, receptor_type='excitatory')
    exc_to_inh = sim.Projection(exc_neurons, inh_neurons, connector, synapse_exc, receptor_type='excitatory')
    inh_to_exc = sim.Projection(inh_neurons, exc_neurons, connector, synapse_inh, receptor_type='inhibitory')

    # Set up recording
    exc_neurons.record(['spikes'])
    inh_neurons.record(['spikes'])

    # Record membrane potentials for the first 'num_recorded' excitatory neurons
    if num_recorded > 0:
        num_recorded = min(num_recorded, num_excitatory)
        v_neurons = exc_neurons[:num_recorded]
        v_neurons.record('v')
    else:
        v_neurons = None

    sim.run(runtime)

    exc_data = exc_neurons.get_data().segments[0].spiketrains
    inh_data = inh_neurons.get_data().segments[0].spiketrains
    if v_neurons is not None:
        membrane_potential = v_neurons.get_data().segments[0].filter(name='v')
    else:
        membrane_potential = None

    sim.end()

    return exc_data, inh_data, membrane_potential

# Streamlit app
def main():
    st.title("Interactive PyNN Simulation")

    # Input parameters
    input_spike_times = st.text_area("Input Spike Times (comma separated)", "10, 20, 30, 40, 50")
    try:
        input_spike_times = list(map(float, input_spike_times.split(',')))
    except ValueError:
        st.error("Invalid input spike times. Please enter comma-separated numbers.")
        return

    num_excitatory = st.number_input("Number of Excitatory Neurons", min_value=1, value=100)
    num_inhibitory = st.number_input("Number of Inhibitory Neurons", min_value=1, value=25)
    num_recorded = st.number_input("Number of Recorded Potentials", min_value=1, max_value=num_excitatory, value=3)

    # Synapse Connectivity Type
    synapse_type = st.selectbox(
        "Synapse Connectivity Type",
        ["AllToAll", "OneToOne", "PairwiseBernoulli", "FixedTotalNumber", "FixedInDegree", "FixedOutDegree"]
    )

    # Synapse Parameters
    st.sidebar.header("Synapse Parameters")
    connector_params = {}

    if synapse_type == 'AllToAll':
        allow_self_connections = st.sidebar.checkbox(
            "Allow Self Connections", value=False, key="allow_self_connections_AllToAll"
        )
        connector_params['allow_self_connections'] = allow_self_connections

    elif synapse_type == 'OneToOne':
        # No allow self connections for OneToOnei
        connector_params['allow_self_connections'] = False

    elif synapse_type == 'PairwiseBernoulli':
        allow_self_connections = st.sidebar.checkbox(
            "Allow Self Connections", value=False, key="allow_self_connections_PairwiseBernoulli"
        )
        p_connect = st.sidebar.slider(
            "Probability of Connection (p_connect)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="p_connect_PairwiseBernoulli"
        )
        connector_params['allow_self_connections'] = allow_self_connections
        connector_params['p_connect'] = p_connect

    elif synapse_type == 'FixedTotalNumber':
        allow_self_connections = st.sidebar.checkbox(
            "Allow Self Connections", value=False, key="allow_self_connections_FixedTotalNumber"
        )
        n = st.sidebar.number_input(
            "Number of Connections (n)", min_value=1, value=10, key="n_FixedTotalNumber"
        )
        connector_params['allow_self_connections'] = allow_self_connections
        connector_params['n'] = n

    elif synapse_type == 'FixedInDegree':
        allow_self_connections = st.sidebar.checkbox(
            "Allow Self Connections", value=False, key="allow_self_connections_FixedInDegree"
        )
        n = st.sidebar.number_input(
            "Number of Incoming Connections (n)", min_value=1, value=10, key="n_FixedInDegree"
        )
        connector_params['allow_self_connections'] = allow_self_connections
        connector_params['n'] = n

    elif synapse_type == 'FixedOutDegree':
        allow_self_connections = st.sidebar.checkbox(
            "Allow Self Connections", value=False, key="allow_self_connections_FixedOutDegree"
        )
        n = st.sidebar.number_input(
            "Number of Outgoing Connections (n)", min_value=1, value=10, key="n_FixedOutDegree"
        )
        connector_params['allow_self_connections'] = allow_self_connections
        connector_params['n'] = n

    # Synapse Weights and Delays
    weight_exc = st.slider(
        "Excitatory Synapse Weight (nS)",
        min_value=0.0, max_value=20.0, value=5.0, step=0.1
    )
    weight_inh = st.slider(
        "Inhibitory Synapse Weight (nS)",
        min_value=0.0, max_value=20.0, value=5.0, step=0.1
    )
    delay_exc = st.slider(
        "Excitatory Synapse Delay (ms)", min_value=0.0, max_value=10.0, value=1.0, step=0.1
    )
    delay_inh = st.slider(
        "Inhibitory Synapse Delay (ms)", min_value=0.0, max_value=10.0, value=1.0, step=0.1
    )

    # Simulation Parameters
    timestep = st.number_input("Timestep (ms)", min_value=0.01, value=0.1)
    runtime = st.number_input("Simulation Run Time (ms)", min_value=1, value=100)

    # Neuron parameters dropdown (allow editing)
    with st.expander("Neuron Parameters"):
        st.write("Edit neuron parameters:")
        v_rest = st.number_input("Resting membrane potential (v_rest) (mV)", value=-65.0)
        v_reset = st.number_input("Reset potential after a spike (v_reset) (mV)", value=-65.0)
        v_thresh = st.number_input("Spike threshold (v_thresh) (mV)", value=-50.0)
        tau_m = st.number_input("Membrane time constant (tau_m) (ms)", value=20.0)
        tau_syn_E = st.number_input("Excitatory synaptic time constant (tau_syn_E) (ms)", value=5.0)
        tau_syn_I = st.number_input("Inhibitory synaptic time constant (tau_syn_I) (ms)", value=5.0)
        cm = st.number_input("Membrane capacitance (cm) (pF)", value=200.0)

    # Define cell_params dictionary
    cell_params = {
        'v_rest': v_rest,
        'v_reset': v_reset,
        'v_thresh': v_thresh,
        'tau_m': tau_m,
        'tau_syn_E': tau_syn_E,
        'tau_syn_I': tau_syn_I,
        'cm': cm
    }

    # Run simulation button
    if st.button("Run Simulation"):
        exc_data, inh_data, membrane_potential = run_simulation(
            input_spike_times, num_excitatory, num_inhibitory,
            synapse_type, weight_exc, weight_inh,
            delay_exc, delay_inh, timestep, runtime,
            cell_params, num_recorded, connector_params
        )

        if exc_data is None:
            st.error("Simulation failed due to invalid synapse type.")
            st.stop()

        # Calculate average spike intensity
        total_spikes_exc = sum(len(spiketrain) for spiketrain in exc_data)
        average_spike_intensity_exc = total_spikes_exc / num_excitatory

        total_spikes_inh = sum(len(spiketrain) for spiketrain in inh_data)
        average_spike_intensity_inh = total_spikes_inh / num_inhibitory

        # Display average spike intensities
        st.write(f"**Average Spike Intensity for Excitatory Neurons:** {average_spike_intensity_exc:.2f} spikes/neuron")
        st.write(f"**Average Spike Intensity for Inhibitory Neurons:** {average_spike_intensity_inh:.2f} spikes/neuron")

        # Calculate firing rates
        exc_time_bins, exc_firing_rate = calculate_firing_rate(exc_data, runtime)
        inh_time_bins, inh_firing_rate = calculate_firing_rate(inh_data, runtime)

        # Plotting firing rates
        plt.figure(figsize=(15, 6))
        plt.plot(exc_time_bins, exc_firing_rate, label='Excitatory Firing Rate', color='blue')
        plt.plot(inh_time_bins, inh_firing_rate, label='Inhibitory Firing Rate', color='red')
        plt.xlabel("Time (ms)")
        plt.ylabel("Firing Rate (spikes/s)")
        plt.title("Firing Rate of Neurons")
        plt.legend()
        plt.xlim(0, runtime)  # Set x-axis to simulation run time
        plt.tight_layout()

        # Display the firing rate plot using st.pyplot
        st.pyplot(plt)
        plt.close()

        # Plotting spike raster
        plt.figure(figsize=(15, 8))  # Set consistent figure size

        # Plotting spike raster
        plt.figure(figsize=(15, 8))  # Set consistent figure size
        for idx, spiketrain in enumerate(exc_data):
            plt.vlines(spiketrain, idx + 0.5, idx + 1.5, color='blue')
        for idx, spiketrain in enumerate(inh_data):
            plt.vlines(spiketrain, idx + 0.5, idx + 1.5, color='red')

        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron Index")
        plt.title("Raster Plot of Neuron Firing Activity")
        plt.xlim(0, runtime)  # Set x-axis to simulation run time
        plt.ylim(0.5, num_excitatory + num_inhibitory + 0.5)  # Adjust y-axis based on neuron counts
        plt.tight_layout()

        # Display the spike raster plot using st.pyplot
        st.pyplot(plt)
        plt.close()

        # Display Spike Times for Excitatory Neurons
        if exc_data and len(exc_data) > 0:
            with st.expander("Spike Times for Excitatory Neurons"):
                for i, spiketrain in enumerate(exc_data):
                    spike_times = sorted([float(spike) for spike in spiketrain])
                    st.write(f"**Neuron {i + 1}:** {spike_times}")

        # Display Spike Times for Inhibitory Neurons
        if inh_data and len(inh_data) > 0:
            with st.expander("Spike Times for Inhibitory Neurons"):
                for i, spiketrain in enumerate(inh_data):
                    spike_times = sorted([float(spike) for spike in spiketrain])
                    st.write(f"**Neuron {i + 1}:** {spike_times}")

        # Plot membrane potentials
        if membrane_potential and len(membrane_potential) > 0:
            plt.figure(figsize=(15, 8))  # Set consistent figure size

            st.header("Membrane Potentials of Recorded Excitatory Neurons")
            for i in range(len(membrane_potential)):
                # Get the time array for the current neuron's membrane potential
                time = membrane_potential[i].times  # This should have the correct shape

                # Check if the membrane potential has multiple columns
                if len(membrane_potential[i].shape) > 1:
                    # Select the first column (assuming it's the main membrane potential)
                    v = membrane_potential[i][:, 0]
                else:
                    v = membrane_potential[i]

                # Ensure the shapes match before plotting
                if len(time) == len(v):
                    # Plot the membrane potential
                    plt.plot(time, v, label=f'Neuron {i + 1}')
                else:
                    st.error(f"Dimension mismatch for neuron {i + 1}: "
                            f"time shape {time.shape}, membrane potential shape {v.shape}")
                    # Optionally, truncate the data to match
                    min_length = min(len(time), len(v))
                    plt.plot(time[:min_length], v[:min_length], label=f'Neuron {i + 1}')

            plt.xlabel("Time (ms)")
            plt.ylabel("Membrane Potential (mV)")
            plt.title("Membrane Potentials of Excitatory Neurons")
            plt.xlim(0, runtime)  # Match x-axis with spike raster
            plt.legend()
            plt.tight_layout()

            # Display the membrane potential plot using st.pyplot
            st.pyplot(plt)
            plt.close()
        else:
            st.write("No membrane potential data available.")

        # Debugging prints to check data structure
        st.write(f"Number of recorded excitatory potentials: {len(membrane_potential) if membrane_potential else 0}")
        if membrane_potential and len(membrane_potential) > 0:
            # Display a sample of the first neuron's membrane potential
            sample = membrane_potential[0][:10, 0] if len(membrane_potential[0].shape) > 1 else membrane_potential[0][:10]
            st.write(f"First neuron's potential sample: {sample} mV")
        else:
            st.write("No membrane potentials recorded.")

if __name__ == "__main__":
    main()
