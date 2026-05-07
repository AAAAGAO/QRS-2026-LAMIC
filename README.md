Information about user studies and application queries can be found in our materials.
Here are the details of our RQ1 in two datasets
**Table 1: PERFORMANCE COMPARISON ON MCGILL DATASET**

<table>
  <thead>
    <tr>
      <th rowspan="2">Approach</th>
      <th colspan="3">JodaTime</th>
      <th colspan="3">Math Library</th>
      <th colspan="3">Col.Official</th>
      <th colspan="3">Col.Jenkov</th>
      <th colspan="3">Smack</th>
    </tr>
    <tr>
      <th>P</th><th>R</th><th>F</th>
      <th>P</th><th>R</th><th>F</th>
      <th>P</th><th>R</th><th>F</th>
      <th>P</th><th>R</th><th>F</th>
      <th>P</th><th>R</th><th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GMR</td>
      <td>59.12</td><td>48.00</td><td>53.24</td>
      <td>43.86</td><td>44.00</td><td>43.58</td>
      <td>47.60</td><td>47.70</td><td>47.40</td>
      <td>51.38</td><td>50.72</td><td>43.38</td>
      <td>43.54</td><td>43.48</td><td>43.17</td>
    </tr>
    <tr>
      <td>FITSEA</td>
      <td>69.61</td><td>66.50</td><td>67.95</td>
      <td>52.53</td><td>51.37</td><td>45.00</td>
      <td>50.20</td><td>50.20</td><td>48.00</td>
      <td>52.42</td><td>51.34</td><td>40.19</td>
      <td>58.24</td><td>58.33</td><td>58.30</td>
    </tr>
    <tr>
      <td>FRAPT</td>
      <td>49.25</td><td>71.74</td><td>58.41</td>
      <td>51.55</td><td>64.84</td><td>57.44</td>
      <td>43.97</td><td>64.49</td><td>52.28</td>
      <td>44.82</td><td>66.41</td><td>53.52</td>
      <td>68.89</td><td>77.50</td><td>72.94</td>
    </tr>
    <tr>
      <td>DTML</td>
      <td>90.55</td><td>82.49</td><td>85.00</td>
      <td>81.39</td><td>80.68</td><td>79.13</td>
      <td>69.03</td><td>65.91</td><td>60.82</td>
      <td>79.45</td><td>70.87</td><td>72.58</td>
      <td>83.23</td><td>81.35</td><td>83.11</td>
    </tr>
    <tr>
      <td>lamic</td>
      <td><strong>91.49</strong></td><td><strong>94.51</strong></td><td><strong>92.97</strong></td>
      <td><strong>88.33</strong></td><td><strong>92.17</strong></td><td><strong>90.21</strong></td>
      <td><strong>78.53</strong></td><td><strong>85.71</strong></td><td><strong>82.02</strong></td>
      <td><strong>68.70</strong></td><td><strong>72.13</strong></td><td><strong>70.37</strong></td>
      <td><strong>87.01</strong></td><td><strong>88.16</strong></td><td><strong>87.90</strong></td>
    </tr>
  </tbody>
</table>

**P: Precision, R: Recall, F: F-Measure.**

---

**Table 2: PERFORMANCE COMPARISON ON ANDROID DATASET**

<table>
  <thead>
    <tr>
      <th rowspan="2">Approach</th>
      <th colspan="3">Graphics</th>
      <th colspan="3">Resources</th>
      <th colspan="3">Text</th>
      <th colspan="3">Data</th>
    </tr>
    <tr>
      <th>P</th><th>R</th><th>F</th>
      <th>P</th><th>R</th><th>F</th>
      <th>P</th><th>R</th><th>F</th>
      <th>P</th><th>R</th><th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GMR</td>
      <td>55.42</td><td>50.08</td><td>47.62</td>
      <td>48.13</td><td>48.01</td><td>47.84</td>
      <td>47.18</td><td>47.72</td><td>47.00</td>
      <td>47.41</td><td>48.38</td><td>41.25</td>
    </tr>
    <tr>
      <td>FITSEA</td>
      <td>62.15</td><td>51.61</td><td>46.00</td>
      <td>50.00</td><td>61.31</td><td>55.18</td>
      <td>50.63</td><td>59.20</td><td>54.54</td>
      <td>50.42</td><td>50.04</td><td>49.61</td>
    </tr>
    <tr>
      <td>FRAPT</td>
      <td>39.32</td><td>59.30</td><td>47.28</td>
      <td>41.58</td><td>59.36</td><td>48.91</td>
      <td>51.66</td><td>81.95</td><td>63.37</td>
      <td>45.19</td><td>62.24</td><td>52.36</td>
    </tr>
    <tr>
      <td>DTML</td>
      <td>81.39</td><td>85.68</td><td>80.93</td>
      <td>91.90</td><td>79.04</td><td>75.14</td>
      <td>81.64</td><td>82.98</td><td>81.53</td>
      <td>91.91</td><td>78.19</td><td>83.43</td>
    </tr>
    <tr>
      <td>lamic</td>
      <td><strong>83.21</strong></td><td><strong>86.75</strong></td><td><strong>82.90</strong></td>
      <td><strong>92.35</strong></td><td><strong>80.12</strong></td><td><strong>77.85</strong></td>
      <td><strong>83.92</strong></td><td><strong>84.11</strong></td><td><strong>82.07</strong></td>
      <td><strong>92.44</strong></td><td><strong>78.05</strong></td><td><strong>84.10</strong></td>
    </tr>
  </tbody>
</table>

**P: Precision, R: Recall, F: F-Measure.**
