import bpy
import bmesh

# Set up VR settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.ops.preferences.addon_enable(module='cycles')
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
#bpy.context.preferences.addons['cycles'].preferences.devices[0].use = True
bpy.ops.wm.xr_session_start()

# Load glTF environment
bpy.ops.import_scene.gltf(filepath='horror.osgb')

# Set environment position and scale
env_obj = bpy.data.objects['horror']
env_obj.location = (0, 0, 0)
env_obj.scale = (1, 1, 1)

# Enter VR mode
bpy.ops.wm.xr_session_enter()

# Run the Blender event loop
def main():
    while True:
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        bpy.ops.wm.xr_frame_push()
        bpy.ops.wm.xr_frame_pull()
        bpy.ops.wm.xr_event_pull()
main()

