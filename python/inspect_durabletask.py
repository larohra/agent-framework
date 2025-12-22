import inspect
import durabletask
from durabletask.task import OrchestrationContext

print("=" * 80)
print("OrchestrationContext methods (for entity operations):")
print("=" * 80)
for name in dir(OrchestrationContext):
    if not name.startswith('_') and 'entity' in name.lower():
        try:
            attr = getattr(OrchestrationContext, name)
            if callable(attr):
                sig = inspect.signature(attr)
                print(f"  {name}{sig}")
        except:
            pass

print("\n" + "=" * 80)
print("Checking OrchestrationState for entity state access:")
print("=" * 80)
try:
    from durabletask.client import OrchestrationState
    for name in dir(OrchestrationState):
        if not name.startswith('_'):
            print(f"  {name}")
except Exception as e:
    print(f"Could not inspect OrchestrationState: {e}")
