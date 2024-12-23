import cProfile
import pstats
import io

from anemone.forwards import *
from test_inversion import test_inversion_ForwardLaplace3


rho = torch.ones(3)*40.
rho_change = torch.linspace(-30, 260, 3)
rho = rho[:, None] + rho_change[None, :]
rho[0, :] = 40.
rho[2, :] = 40.
thick = torch.ones_like(rho)[:-1, :]*20
thick =torch.ones_like(rho[:-1])*20




def your_function_to_profile(fwr):
    dbdt = fwr(rho, thick)

def convert_to_markdown(profiling_output):
    lines = profiling_output.split('\n')
    header = "| Function | Calls | Total Time | Per Call | Cumulative Time | Per Call |"
    separator = "| --- | --- | --- | --- | --- | --- |"
    markdown_lines = [header, separator]
    
    for line in lines[5:]:  # Skip the first few lines of the output
        if line.strip() == "":
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        function = " ".join(parts[5:]).replace('c:\\users\\au646858\\onedrive - aarhus universitet\\documents\\_repositories\\', '')
        calls = parts[0]
        total_time = parts[2]
        per_call = parts[3]
        cumulative_time = parts[4]
        per_call_cumulative = parts[1]
        markdown_line = f"| {function} | {calls} | {total_time} | {per_call} | {cumulative_time} | {per_call_cumulative} |"
        markdown_lines.append(markdown_line)
    return "\n".join(markdown_lines)
    

if __name__ == "__main__":
    with open("C:/Users/au646858/notes/profiling_results.md", 'w') as f:
        f.write("# Profiling results\n")

        for fwrmaker in [ForwardLaplace3]:
            f.write('## ' + fwrmaker.__name__ + '\n')
            f.write('### initial_run \n')
            fwr = fwrmaker.example()

            pr = cProfile.Profile()
            pr.enable()
            
            your_function_to_profile(fwr)
            
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            profiling_output = s.getvalue()
            
            markdown_output = convert_to_markdown(profiling_output)
            f.write(markdown_output+'\n\n')
            f.write('### run 2 \n')

            pr = cProfile.Profile()
            pr.enable()
            
            your_function_to_profile(fwr)
            
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            profiling_output = s.getvalue()
            
            markdown_output = convert_to_markdown(profiling_output)
            f.write(markdown_output+'\n\n')