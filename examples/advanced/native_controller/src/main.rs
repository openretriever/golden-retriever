//! Rust IK Solver Node for Retriever
//!
//! Receives TargetPose (x,y,z,r,p,y) and computes JointAngles.

use dora_node_api::{self, dora_core::config::DataId, DoraNode, Event};
use dora_node_api::arrow::array::{Array, BinaryArray};
use eyre::{Context, Result};

fn main() -> Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, .. } => {
                if id.as_str() == "pose" {
                    let pose = deserialize_f32_array(data)?;
                    
                    if pose.len() >= 6 {
                        // Mock IK Calculation
                        let x = pose[0];
                        let y = pose[1];
                        let z = pose[2];
                        
                        let j1 = y.atan2(x);
                        let j2 = z * 2.0;
                        let j3 = x + y;
                        
                        let joints = vec![j1, j2, j3, 0.0, 0.0, 0.0];
                        
                        // Serialize output
                        let output_bytes = serialize_f32_vec(&joints);
                        let array = BinaryArray::from_vec(vec![&output_bytes[..]]);
                        
                        node.send_output(DataId::from("joints".to_string()), Default::default(), array)
                            .context("Failed to send joints")?;
                    }
                }
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}

fn deserialize_f32_array(data: dora_node_api::ArrowData) -> Result<Vec<f32>> {
    use dora_node_api::arrow::array::{Float32Array, BinaryArray};

    // Native Float32Array
    if let Some(arr) = data.as_any().downcast_ref::<Float32Array>() {
        return Ok(arr.values().to_vec());
    }

    // Numpy-serialized BinaryArray
    if let Some(arr) = data.as_any().downcast_ref::<BinaryArray>() {
        if arr.len() > 0 {
            let bytes = arr.value(0);
            // Assuming f32 (4 bytes)
            let count = bytes.len() / 4;
            let mut values = Vec::with_capacity(count);
            for i in 0..count {
                let start = i * 4;
                let val = f32::from_le_bytes([
                    bytes[start], bytes[start+1], bytes[start+2], bytes[start+3]
                ]);
                values.push(val);
            }
            return Ok(values);
        }
    }

    eyre::bail!("Could not deserialize f32 array")
}

fn serialize_f32_vec(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}
